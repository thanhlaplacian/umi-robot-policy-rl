"""Batched RL env for "pick up the packaged garment on the shelf" on laplacian-gym's flex garments.

Built on the gym's ``build_jeans_scene`` (livinglab_hq_v2 + LPR1 + 1..5 stacked flex garments on a
rack shelf) and the same UMI action interface as ``LaplacianRLEnv``: per-step EE-local deltas in the
``lpr1/mobile/base`` frame, CPU DLS-IK, position servos, 15 Hz control (267 physics substeps at 4 kHz).

Reset writes a full ``[N, nq]`` qpos: the ready keyframe, the mobile base placed from the gym's
randomization ranges (rack A shelf 3 side-pick profile), and the garment stack jittered per env by
translating/rotating every flex vertex (a port of ``jeans_stack.jitter_stack``). Reward is staged:
reach -> pad contact -> both pads -> lift; success = both rubber pads on the garment and its
centroid >= ``lift_height`` above the post-settle height for ``hold_steps`` consecutive steps
(``garment_success.stable_garment_hold`` without the scene-contact clauses).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from umi_rl.envs.laplacian.rl_env import ARM_JOINTS, CAMERAS, GRIP_CLOSED_CTRL, LaplacianRLEnv

PROMPTS = {
    "left": "Pick up the garment on the shelf. the left arm grasps and the right arm does nothing",
    "right": "Pick up the garment on the shelf. the left arm does nothing and the right arm grasps",
}
TARGET = "packaged_jeans_1"  # the gym names the target flex like this for every SKU


@dataclass
class GarmentRewardConfig:
    beta: float = 5.0               # phi(d) = 1 - tanh(beta d)
    reach_weight: float = 2.0       # stage 0: reach_weight * phi(TCP-to-garment distance)
    pad_stage: float = 3.0          # stage 1: one rubber pad touches the garment
    grasp_stage: float = 4.0        # stage 2: both pads touch; + lift_weight * clip(rise / lift_height)
    lift_weight: float = 4.0
    lift_height: float = 0.07       # metres of centroid rise that count as lifted (gym: > 70 mm)
    lifted_stage: float = 8.0       # stage 3: lifted; + 2 * clip(rise / 0.15)
    success: float = 13.0
    hold_steps: int = 3             # consecutive lifted steps for success
    idle_arm_penalty: float = 0.05  # w * ||q_idle - q_home||^2 on the non-grasping arm
    lower_contact_penalty: float = 0.5  # per step while any robot geom touches a lower garment
    fail_on_fall: bool = True       # terminate when the garment drops off the shelf
    fall_penalty: float = 2.0
    drop_margin: float = 0.05       # centroid this far below its settled height = fell


@dataclass
class GarmentRLConfig:
    scene: str = "configs/livinglab_hq_v2.yaml"   # relative to the laplacian-gym checkout
    shelf: str = "scan/rack_a_shelf_3"
    sku: str = "packaged_jeans_1"                  # packaged_jeans_1 | packaged_dark_jeans_1 | packaged_sweatpants_1
    stack_count: int = 1
    arm: str = "left"
    along: float = -0.20
    overhang: float = -0.02
    yaw_offset_deg: float = 0.0                    # 0 = side seam toward the aisle (gym side-pry default), 90 = quarter turn
    base_distance: tuple[float, float] = (0.54, 0.56)   # gym A3 profile ranges
    base_lateral: tuple[float, float] = (-0.09, -0.07)
    base_yaw_deg: tuple[float, float] = (-1.0, 1.0)
    lift_stage: float | None = None
    jitter_m: float = 0.02
    jitter_yaw_deg: float = 5.0
    arm_noise_rad: float = 0.0                     # >0: uniform joint noise on the grasping arm at reset
    cameras: tuple[str, ...] = ("cam_wrist_left", "cam_wrist_right", "cam_head")
    wrist_hw: tuple[int, int] | None = None        # None = native fisheye 320x240 (the W40 lpr1 datasets)
    control_hz: float = 15.0
    physics_hz: int = 4000
    settle_s: float = 1.0
    max_episode_steps: int = 225                   # 15 s
    ik_iters: int = 4
    ik_damping: float = 0.05
    gripper_mode: str = "linear"                   # linear | binary | boost
    gripper_thresh: float = 0.3
    prompt: str | None = None                      # None = PROMPTS[arm]
    ibl: bool = True                               # image-based lighting from the splat (gym default)
    cull_radius: float = 1.5                       # >0: the garment only collides with the grasping hand and scan geoms within this radius (m) of its spawn
    solver_iterations: int | None = None           # override <option iterations> (scene default 30) for throughput experiments
    ls_iterations: int | None = None               # override <option ls_iterations> (scene default 50)
    seed: int = 0
    reward: GarmentRewardConfig = field(default_factory=GarmentRewardConfig)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = GarmentRewardConfig(**self.reward)
        if self.arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if self.prompt is None:
            self.prompt = PROMPTS[self.arm]


def _cull_flex_collisions(xml_path: Path, garment_pos, arm: str, radius: float, iterations=None, ls_iterations=None) -> Path:
    """Derive a scene where the flex garment collides only with the grasping hand and nearby scan geoms.

    The composed livinglab_hq_v2 scene has ~23k scan collision boxes and 56 robot meshes; mujoco-warp
    tests the flex against all of them every 0.25 ms substep (measured 30 ms/substep). Collision bit 2 is
    given to the flex, to the geoms of ``lpr1/openarm_<arm>_hand*`` bodies and to scan geoms within
    ``radius`` of the garment spawn; everything else keeps bit 1 only, so robot/scene collisions are
    unchanged. Written next to the gym's cached scene, content-addressed by the inputs.
    """
    import hashlib
    import xml.etree.ElementTree as ET

    tag = f"{radius:g}_{hashlib.sha1(np.asarray(garment_pos, dtype=np.float64).round(3).tobytes()).hexdigest()[:8]}"
    if iterations is not None or ls_iterations is not None:
        tag += f"_it{iterations}_ls{ls_iterations}"
    out = xml_path.with_name(f"{xml_path.stem}_cull_{arm}_{tag}.xml")
    if out.exists() and out.stat().st_mtime >= xml_path.stat().st_mtime:
        return out
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for fc in root.iter("flexcomp"):
        ct = fc.find("contact")
        if ct is None:
            ct = ET.SubElement(fc, "contact")
        ct.set("contype", "2"); ct.set("conaffinity", "2")
    g0 = np.asarray(garment_pos, dtype=np.float64)
    n_hand = n_near = n_far = 0

    def bump(geom):
        geom.set("contype", str(int(geom.get("contype", "1")) | 2)); geom.set("conaffinity", str(int(geom.get("conaffinity", "1")) | 2))

    def walk(body, origin, hand):
        nonlocal n_hand, n_near, n_far
        for child in body:
            if child.tag == "geom":
                if child.get("contype", "1") == "0" and child.get("conaffinity", "1") == "0":
                    continue
                if hand:
                    bump(child); n_hand += 1
                elif (child.get("name") or "").startswith("scan/") or body.tag == "worldbody":
                    pos = origin + np.fromstring(child.get("pos", "0 0 0"), sep=" ")
                    if np.linalg.norm(pos - g0) <= radius:
                        bump(child); n_near += 1
                    else:
                        n_far += 1
            elif child.tag == "body":
                name = child.get("name", "")
                walk(child, origin + np.fromstring(child.get("pos", "0 0 0"), sep=" "), hand or f"openarm_{arm}_hand" in name)

    walk(root.find("worldbody"), np.zeros(3), False)
    opt = root.find("option")
    if opt is not None:
        if iterations is not None:
            opt.set("iterations", str(int(iterations)))
        if ls_iterations is not None:
            opt.set("ls_iterations", str(int(ls_iterations)))
    tree.write(out, encoding="unicode")
    print(f"[garment_env] culled scene: {n_hand} hand geoms + {n_near} nearby scan geoms collide with the garment; {n_far} scan geoms excluded -> {out.name}")
    return out


def _patch_ibl_dtype():
    """livinglab_hq_v2's IBL probe mixes float64 and float32 tensors in ``SplatIBL.__init__``
    (``expected scalar type Double but found Float``); cast the two einsum operands to float32."""
    import laplacian_gym.rendering.lighting as L
    if getattr(L, "_umi_f32", False):
        return
    sh, srgb = L.sh_basis, L.srgb_to_linear
    L.sh_basis = lambda d: sh(d.float() if torch.is_tensor(d) else d).float()
    L.srgb_to_linear = lambda x: srgb(x.float() if torch.is_tensor(x) else x)
    L._umi_f32 = True


class GarmentRLEnv(LaplacianRLEnv):
    """Same step/observe/IK machinery as ``LaplacianRLEnv``; scene, reset and reward are garment specific."""

    def __init__(self, cfg: GarmentRLConfig, num_envs: int = 1, device: str = "cuda:0", render: bool = True, gym_root: str | None = None):
        import laplacian_gym
        from laplacian_gym.env import LaplacianEnv
        from laplacian_gym.tasks.jeans import build_jeans_scene
        from laplacian_gym.tasks.jeans_stack import flex_slice

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        root = Path(gym_root) if gym_root else Path(laplacian_gym.__file__).resolve().parents[2]
        rack_b = self.cfg.shelf.startswith("scan/rack_b_")
        self.face_sign = -1.0 if rack_b else 1.0
        yaw = (90.0 if rack_b else -90.0) + cfg.yaw_offset_deg + (180.0 if cfg.arm == "right" else 0.0)
        scene_cfg, self.meta = build_jeans_scene(
            str(root / cfg.scene), sku=cfg.sku, shelf=cfg.shelf, along=cfg.along, overhang=cfg.overhang,
            yaw_degrees=yaw, stack_count=cfg.stack_count, arm=cfg.arm, physics_hz=cfg.physics_hz)
        if cfg.cull_radius > 0:
            from dataclasses import replace
            scene_cfg = replace(scene_cfg, mjcf=_cull_flex_collisions(Path(scene_cfg.mjcf), self.meta["position"], cfg.arm, cfg.cull_radius, cfg.solver_iterations, cfg.ls_iterations))
        if cfg.ibl:
            _patch_ibl_dtype()
        self.env = LaplacianEnv(scene_cfg, num_envs=num_envs, device=device, render=render, ibl=cfg.ibl)
        self.p = self.env.physics
        self.model = self.p.cpu_model
        m = self.model
        self.k_active = 0 if cfg.arm == "right" else 1  # ARM_JOINTS / tips are right-then-left
        self.tip_ids = [m.body(f"lpr1/openarm_{s}_tcp").id for s in ("right", "left")]
        self.base_body = m.body("lpr1/mobile/base").id
        self.arm_joint_ids = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dofadr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_range = np.stack([m.jnt_range[j] for j in self.arm_joint_ids])
        names = list(self.env.action_names)
        self.arm_act = torch.tensor([names.index(n) for n in ARM_JOINTS], device=self.device)
        self.grip_act = torch.tensor([names.index(f"lpr1/openarm_{s}_hand_base_link__link_1_A") for s in ("right", "left")], device=self.device)
        self.grip_qadr = torch.tensor([m.joint(f"lpr1/openarm_{s}_hand_base_link__link_1_A").qposadr[0] for s in ("right", "left")], device=self.device)
        self.grip_body = [int(m.joint(f"lpr1/openarm_{s}_hand_gripper_joint").bodyid[0]) for s in ("right", "left")]
        self.mobile_qadr = {n: int(m.joint(f"lpr1/mobile/{n}").qposadr[0]) for n in ("x", "y", "yaw", "lift_middle", "lift_top")}
        self.lift_range = m.jnt_range[m.joint("lpr1/mobile/lift_middle").id]
        self._check = mujoco.MjData(m)
        # rest state (keyframe 0 = ready + garment rest pose) for placement geometry and flex rest points
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        gid = m.geom(cfg.shelf).id
        self.shelf_axes = d.geom_xmat[gid].reshape(3, 3).copy()
        self.shelf_centre = d.geom_xpos[gid].copy()
        self.shelf_size = m.geom_size[gid].copy()
        self.shelf_top = float(d.geom_xpos[gid, 2] + m.geom_size[gid, 2])
        placement = m.body("lpr1/placement").id
        self.base_axes = d.xmat[placement].reshape(3, 3).copy()
        self.placement_xy = d.xpos[placement, :2].copy()
        self.garments = list(self.meta["lower_objects"]) + [TARGET]  # bottom -> top
        self.flex_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, n) for n in self.garments}
        self.slices = {n: flex_slice(m, n) for n in self.garments}
        self.rest_points = {n: d.flexvert_xpos[self.slices[n]].copy() for n in self.garments}
        self.origin_rot = {n: d.xmat[m.body(f"{n}/origin").id].reshape(3, 3).copy() for n in self.garments}
        self.vert_qadr = {}
        for n in self.garments:
            bids = m.flex_vertbodyid[self.slices[n]]
            adr = []
            for b in bids:
                if m.body_jntnum[b] != 3:
                    raise ValueError("garment vertices must be unpinned (3 slide joints)")
                j0 = m.body_jntadr[b]
                adr.append([int(m.jnt_qposadr[j0 + k]) for k in range(3)])
            self.vert_qadr[n] = np.array(adr)  # [105, 3]
        # contact lookups
        pad = np.full(m.ngeom, -1, dtype=np.int64)
        robot = np.zeros(m.ngeom, dtype=bool)
        for b in range(m.nbody):
            name = m.body(b).name
            sl = slice(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
            if name.startswith("lpr1/"):
                robot[sl] = True
            for k, letter in enumerate("AB"):
                if name == f"lpr1/openarm_{cfg.arm}_hand_pad_{letter}":
                    pad[sl] = k
        self.pad_of_geom = torch.from_numpy(pad).to(self.device)
        self.robot_geom = torch.from_numpy(robot).to(self.device)
        self.q_home = self.p.default_qpos[0, torch.from_numpy(self.arm_qadr).to(self.device)].reshape(2, 7).clone()
        self.last_reward_terms = {}
        self.rigs = {}
        if render:
            for name in cfg.cameras:
                self.rigs[name] = self._attach_camera(CAMERAS[name], cfg.wrist_hw if "wrist" in name else None)
        self.frame_steps = int(round(cfg.physics_hz / cfg.control_hz))  # 267 at 4 kHz / 15 Hz
        self.elapsed = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.rng = np.random.default_rng(cfg.seed)
        self.timing = {}
        n_v = self.slices[TARGET].stop - self.slices[TARGET].start
        self._init_points = torch.zeros(num_envs, n_v, 3, device=self.device)
        self._init_z = torch.zeros(num_envs, device=self.device)
        self._hold = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------ reset
    def _base_qpos(self, rng, qpos_row: np.ndarray):
        c = self.cfg
        bd, bl, by = rng.uniform(*c.base_distance), rng.uniform(*c.base_lateral), rng.uniform(*c.base_yaw_deg)
        pos = np.asarray(self.meta["position"])
        desired = (pos - self.shelf_axes[:, 1] * self.face_sign * bd + self.shelf_axes[:, 0] * self.face_sign * bl)[:2]
        slides = np.linalg.solve(self.base_axes[:2, :2], desired - self.placement_xy)
        yaw = (np.arctan2(self.face_sign * self.shelf_axes[1, 1], self.face_sign * self.shelf_axes[0, 1])
               + np.deg2rad(by) - np.arctan2(self.base_axes[1, 0], self.base_axes[0, 0]))
        lift = c.lift_stage if c.lift_stage is not None else max(0.0, 0.14 + (self.shelf_top - 0.226) / 2 + 0.016 * (c.stack_count - 1))
        if c.lift_stage is None and c.shelf.endswith("rack_a_shelf_3"):
            lift = max(lift, 0.15)
        lift = float(np.clip(lift, *self.lift_range))
        for n, v in zip(("x", "y", "yaw", "lift_middle", "lift_top"), (slides[0], slides[1], yaw, lift, lift)):
            qpos_row[self.mobile_qadr[n]] = v

    def _garment_qpos(self, rng, qpos_row: np.ndarray):
        """Port of jeans_stack.jitter_stack: shelf-plane offset + yaw per garment, written as vertex slide qpos."""
        c = self.cfg
        m = self.model
        nominal = (np.asarray(self.meta["position"]) - self.shelf_centre) @ self.shelf_axes
        for n in self.garments:
            for _ in range(1000):
                ang = rng.uniform(-c.jitter_yaw_deg, c.jitter_yaw_deg) if c.jitter_yaw_deg else 0.0
                yaw = np.deg2rad(self.meta["yaw_degrees"] + ang)
                ext = np.abs([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [0.15, 0.20]
                delta = rng.uniform(-c.jitter_m, c.jitter_m, 2) if c.jitter_m else np.zeros(2)
                if np.all(np.abs(nominal[:2] + delta) + ext <= self.shelf_size[:2] - 0.005):
                    break
            else:
                raise ValueError("no in-shelf garment jitter fits this spawn")
            world_delta = self.shelf_axes @ np.r_[delta, 0.0]
            pts = self.rest_points[n]
            pivot = pts.mean(0)
            k = self.shelf_axes[:, 2] * np.deg2rad(ang)
            th = np.linalg.norm(k)
            if th > 1e-9:
                kx = k / th
                K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
                turn = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
            else:
                turn = np.eye(3)
            shift = (pts - pivot) @ turn.T + pivot + world_delta - pts  # [105, 3] world
            local = shift @ self.origin_rot[n]  # R^T @ shift per row
            qpos_row[self.vert_qadr[n]] += local

    @torch.no_grad()
    def reset(self, mask: torch.Tensor | None = None, seeds: list[int] | None = None) -> dict:
        N = self.num_envs
        if mask is None:
            mask = torch.ones(N, device=self.device, dtype=torch.bool)
        qpos = self.p.default_qpos.clone()
        rows = qpos.cpu().numpy()
        for i in torch.nonzero(mask, as_tuple=True)[0].tolist():
            rng = np.random.default_rng(seeds[i] if seeds is not None else self.rng.integers(2**31))
            row = rows[i]
            self._base_qpos(rng, row)
            self._garment_qpos(rng, row)
            if self.cfg.arm_noise_rad > 0:
                adr = self.arm_qadr[7 * self.k_active : 7 * self.k_active + 7]
                row[adr] = np.clip(row[adr] + rng.uniform(-self.cfg.arm_noise_rad, self.cfg.arm_noise_rad, 7),
                                   self.arm_range[7 * self.k_active : 7 * self.k_active + 7, 0],
                                   self.arm_range[7 * self.k_active : 7 * self.k_active + 7, 1])
        qpos = torch.from_numpy(rows).to(self.device)
        self.env.reset(mask, qpos=qpos)
        # settle in control-frame chunks: WarpPhysics captures one CUDA graph per distinct substep
        # count, and a 4000-substep graph takes many minutes to capture; the 267-step graph is the
        # one every control step reuses anyway.
        n = int(round(self.cfg.settle_s * self.cfg.physics_hz / self.frame_steps))
        ctrl = self.p.ctrl.clone()
        for _ in range(n):
            self.p.step(ctrl, self.frame_steps)
        pts = self.p.flex_position[:, self.slices[TARGET]]
        self._init_points[mask] = pts[mask]
        self._init_z[mask] = pts[mask, :, 2].mean(-1)
        self._hold[mask] = 0
        self.elapsed = torch.where(mask, torch.zeros_like(self.elapsed), self.elapsed)
        for rig in self.rigs.values():
            if rig["frame"] is not None:
                rig["frame"].valid[mask] = False
        return self.observe()

    # ------------------------------------------------------------------ garment state / reward
    @torch.no_grad()
    def garment_points(self, name: str = TARGET) -> torch.Tensor:
        return self.p.flex_position[:, self.slices[name]]  # [N, 105, 3]

    @torch.no_grad()
    def object_positions(self) -> torch.Tensor:
        return self.garment_points().mean(1)[:, None]  # [N, 1, 3] centroid (episode logger)

    @torch.no_grad()
    def min_tcp_object_distance(self) -> torch.Tensor:
        tcp = self.p.body_pos[:, self.tip_ids[self.k_active]]  # [N, 3]
        return torch.linalg.vector_norm(self.garment_points() - tcp[:, None], dim=-1).amin(-1)

    @torch.no_grad()
    def pad_contacts(self):
        """(pads [N, 2] bool: rubber pad A/B of the grasping arm touches the target, lower [N] bool: robot touches a lower garment)."""
        c = self.p.contacts()
        flex = c["flex"].long()
        g = c["geom"].long().amax(-1)  # the non-flex geom of a flex contact
        valid = c["valid"] & (g >= 0)
        on_target = valid & (flex == self.flex_ids[TARGET]).any(-1)
        pad = self.pad_of_geom[g.clamp(min=0)]
        pads = torch.stack([(on_target & (pad == k)).any(-1) for k in range(2)], -1)
        lower = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for n in self.garments[:-1]:
            lower |= (valid & (flex == self.flex_ids[n]).any(-1) & self.robot_geom[g.clamp(min=0)]).any(-1)
        return pads, lower

    def compute_reward(self):
        R = self.cfg.reward
        N = self.num_envs
        phi = lambda d: 1.0 - torch.tanh(R.beta * d)
        pts = self.garment_points()
        rise = pts[:, :, 2].mean(-1) - self._init_z
        d = self.min_tcp_object_distance()
        pads, lower = self.pad_contacts()
        n_pads = pads.sum(-1)
        lifted = (n_pads == 2) & (rise >= R.lift_height)
        self._hold = torch.where(lifted, self._hold + 1, torch.zeros_like(self._hold))
        success = self._hold >= R.hold_steps
        stage = R.reach_weight * phi(d)
        stage = torch.where(n_pads == 1, torch.full_like(stage, R.pad_stage), stage)
        stage = torch.where(n_pads == 2, R.grasp_stage + R.lift_weight * (rise / R.lift_height).clamp(0, 1), stage)
        stage = torch.where(lifted, R.lifted_stage + 2.0 * (rise / 0.15).clamp(0, 1), stage)
        stage = torch.where(success, torch.full_like(stage, R.success), stage)
        reward = stage
        terms = {"stage": stage, "pads": n_pads.float(), "grasp": (n_pads == 2).float(), "lifted": lifted.float(), "rise": rise, "d_reach": d, "aligned": (d < 0.03).float()}
        if R.idle_arm_penalty > 0:
            q = self.p.qpos[:, torch.from_numpy(self.arm_qadr).to(self.device)].reshape(N, 2, 7)
            dev = ((q - self.q_home[None]) ** 2).sum(-1)[:, 1 - self.k_active]
            reward = reward - R.idle_arm_penalty * dev
            terms["idle_pen"] = R.idle_arm_penalty * dev
        if R.lower_contact_penalty > 0 and len(self.garments) > 1:
            pen = R.lower_contact_penalty * lower.float()
            reward = reward - pen
            terms["disturb_pen"] = pen
        self.last_reward_terms = terms
        return reward, success

    @torch.no_grad()
    def fallen(self) -> torch.Tensor:
        z = self.garment_points()[:, :, 2].mean(-1)
        blown = ~torch.isfinite(self.p.qpos).all(-1) | ~torch.isfinite(z)
        return (z < self._init_z - self.cfg.reward.drop_margin) | blown

    @torch.no_grad()
    def in_basket(self, o):  # no basket in this task
        return torch.zeros(o.shape[:-1], dtype=torch.bool, device=self.device)
