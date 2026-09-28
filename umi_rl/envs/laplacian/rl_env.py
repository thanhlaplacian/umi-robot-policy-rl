"""Batched RL environment on laplacian-gym for the UMI bimanual policy.

Bypasses the gym's planner-driven ``PickPlaceTask`` and drives ``LaplacianEnv`` directly:

* ``reset`` samples object poses on the shelf with the same bands as the data generator
  (``tasks/pick_place.py::spawn``), keeps the robot base at the nominal pose (layout ``center``)
  and settles physics for ``settle_s`` seconds;
* ``step`` takes the policy's per-step action ``[N, 14]`` (grip_R, EE-local dxyz_R, EE-local
  rotvec_R, grip_L, dxyz_L, rotvec_L), integrates the EE-local delta on the current TCP pose,
  solves joint targets with a damped least-squares IK on MuJoCo's Jacobian (CPU, per env),
  writes position-servo targets and runs one 15 Hz frame (40 physics substeps);
* observations: the three policy cameras rendered at 640x480 (fisheye wrists scaled from the
  imported calibration; head pinhole at its native 640x400), plus the 14-dim state in the robot
  base frame (grippers normalized 0..1, poses as xyz + axis-angle), i.e. ``umi_bimanual_v2``;
* reward: sparse success when every object rests inside the basket AABB used by the data
  generator, plus an optional dense term on TCP-to-object distance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .rotations import mat_to_rotvec, rotvec_to_mat

CAMERAS = {"cam_wrist_left": "lpr1/openarm_left_wrist_camera", "cam_wrist_right": "lpr1/openarm_right_wrist_camera", "cam_head": "lpr1/cam_head"}
GRIP_CLOSED_CTRL = -1.0372  # gym's closed gripper command; 0 = open
ARM_JOINTS = [f"lpr1/openarm_{side}_joint{j}" for side in ("right", "left") for j in range(1, 8)]


@dataclass
class RewardConfig:
    """Per-term switches of the reward (docs/reward catalog artifact, section "Proposed reward").

    All terms are additive except the staged term, which overwrites stage bases PlaceSphere-style.
    0.0 disables a weighted term; ``staged=False`` keeps the legacy sparse(+reach) reward.
    """
    staged: bool = False            # items 1+3: reach/grasp/lift/place/release stages with gating
    release_bonus: float = 2.0      # item 2: coefficient of the acting gripper's opening once the object is in the basket
    idle_arm_penalty: float = 0.0   # item 4: w * ||q_idle - q_home||^2
    disturb_penalty: float = 0.0    # item 4: w * sum_j tanh(5 * ||o_j - o_j0||) over non-active, not-yet-placed objects
    regress_penalty: float = 0.0    # item 5: one-shot -w when an object leaves the basket after being inside
    success: float = 13.0           # reward written on the success step (all objects in the basket)
    reach_weight: float = 0.0       # legacy dense term: -w * min TCP-object distance (only when staged=False)
    completion_bonus: float = 10.0  # added per object already in the basket (staged only)
    beta: float = 5.0               # length scale of phi(d) = 1 - tanh(beta d)
    grasp_close_thresh: float = 0.3 # gripper closed fraction above which a finger contact counts as a grasp
    lift_height: float = 0.025      # metres above the reset height that count as lifted
    basket_target_z: float = 0.05   # target point = basket centre XY, rim height + this
    fail_on_fall: bool = True       # terminate with fall_penalty when an object drops below the shelf top
    fall_penalty: float = 2.0
    fall_margin: float = 0.06       # metres below the reset height that count as fallen


@dataclass
class LaplacianRLConfig:
    scene: str = "configs/livinglab_hq.yaml"  # relative to the laplacian-gym checkout
    spawn_count: int = 2
    object_types: tuple[str, ...] = ("ROB-001", "ROB-003", "RTC-001")
    cameras: tuple[str, ...] = ("cam_wrist_left", "cam_wrist_right", "cam_head")
    wrist_hw: tuple[int, int] = (480, 640)
    settle_s: float = 0.5
    max_episode_steps: int = 300  # 15 Hz frames = 20 s
    ik_iters: int = 4
    ik_damping: float = 0.05
    dense_reward: float = 0.0     # legacy alias of reward.reach_weight
    success_reward: float = 1.0   # legacy alias of reward.success when reward.staged is False
    seed: int = 0
    randomization: str = "baseline"  # object poses only; base stays nominal
    reward: RewardConfig = field(default_factory=RewardConfig)
    # policy gripper g in [0,1] -> servo command. "linear": -1.0372*g (half-closed for g=0.5, too weak to hold);
    # "binary": fully closed when g > gripper_thresh (the data generator's convention); "boost": -1.0372*min(1, g/gripper_thresh).
    gripper_mode: str = "linear"
    gripper_thresh: float = 0.3
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)
        if self.dense_reward > 0 and self.reward.reach_weight == 0:
            self.reward.reach_weight = self.dense_reward
        if not self.reward.staged and self.success_reward != 1.0 and self.reward.success == 13.0:
            self.reward.success = self.success_reward
        elif not self.reward.staged and self.reward.success == 13.0:
            self.reward.success = self.success_reward


class LaplacianRLEnv:
    def __init__(self, cfg: LaplacianRLConfig, num_envs: int, device: str = "cuda:0", gym_root: str | None = None, render: bool = True):
        import laplacian_gym
        from laplacian_gym.env import LaplacianEnv
        from laplacian_gym.tasks.scene import BASE, BASKET_RIM, DIMENSIONS, SHELF_TOP, build_pick_place_scene
        from pathlib import Path

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        root = Path(gym_root) if gym_root else Path(laplacian_gym.__file__).resolve().parents[2]
        scene_cfg, self.slots = build_pick_place_scene(str(root / cfg.scene), cfg.spawn_count, tuple(cfg.object_types))
        self.env = LaplacianEnv(scene_cfg, num_envs=num_envs, device=device, render=render)
        self.p = self.env.physics
        self.model = self.p.cpu_model
        m = self.model
        self._dims, self._shelf_top, self._basket_rim, self._base = DIMENSIONS, SHELF_TOP, BASKET_RIM, np.array(BASE)
        # bodies / joints
        self.tip_ids = [m.body(f"lpr1/openarm_{s}_tcp").id for s in ("right", "left")]
        self.base_body = m.body("lpr1/placement").id
        self.arm_joint_ids = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dofadr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_range = np.stack([m.jnt_range[j] for j in self.arm_joint_ids])
        names = list(self.env.action_names)
        self.arm_act = torch.tensor([names.index(n) for n in ARM_JOINTS], device=self.device)
        self.grip_act = torch.tensor([names.index(f"lpr1/openarm_{s}_hand_base_link__link_1_A") for s in ("right", "left")], device=self.device)
        self.grip_qadr = torch.tensor([m.joint(f"lpr1/openarm_{s}_hand_base_link__link_1_A").qposadr[0] for s in ("right", "left")], device=self.device)
        self.object_bodies = [m.body(slot["body"]).id for slot in self.slots]  # active variant per slot, updated at reset
        self._check = mujoco.MjData(m)
        # geom -> object slot / gripper arm lookups for contact-based grasp detection
        obj_of = np.full(m.ngeom, -1, dtype=np.int64)
        for j, slot in enumerate(self.slots):
            for v in slot.get("variants", [slot]):
                b = m.body(v["body"]).id
                obj_of[m.body_geomadr[b] : m.body_geomadr[b] + m.body_geomnum[b]] = j
        arm_of = np.full(m.ngeom, -1, dtype=np.int64)
        for b in range(m.nbody):
            name = m.body(b).name
            for k, side in enumerate(("right", "left")):
                if f"openarm_{side}_hand" in name:
                    arm_of[m.body_geomadr[b] : m.body_geomadr[b] + m.body_geomnum[b]] = k
        self.obj_of_geom = torch.from_numpy(obj_of).to(self.device)
        self.arm_of_geom = torch.from_numpy(arm_of).to(self.device)
        self.q_home = self.p.default_qpos[0, torch.from_numpy(self.arm_qadr).to(self.device)].reshape(2, 7).clone()  # ready keyframe per arm
        self.basket_target = torch.tensor([*__import__("laplacian_gym.tasks.scene", fromlist=["BASKET_CENTER"]).BASKET_CENTER, self._basket_rim + cfg.reward.basket_target_z], device=self.device)
        self.last_reward_terms = {}
        # cameras
        self.rigs = {}
        if render:
            for name in cfg.cameras:
                self.rigs[name] = self._attach_camera(CAMERAS[name], cfg.wrist_hw if "wrist" in name else None)
        self.frame_steps = scene_cfg.physics_hz // scene_cfg.render_hz  # 40
        self.elapsed = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.rng = np.random.default_rng(cfg.seed)
        self.timing = {}

    # ------------------------------------------------------------------ cameras
    def _attach_camera(self, name, hw):
        """Like ``LaplacianEnv.attach_camera`` but at the policy's resolution (K scaled)."""
        import ast, json
        from laplacian_gym.rendering.camera import CameraBatch
        from laplacian_gym.rendering.fisheye import FisheyeCameraBatch

        env = self.env
        cid = self.model.camera(name).id
        root = env.config.splat.parents[2]
        meta = json.loads((root / "robots" / env.config.robot / "mjcf/conversion.json").read_text())
        data = next(c for c in meta["cameras"] if c["name"] == name.split("/")[-1])["attributes"]
        prefix = "omni:lensdistortion:opencvFisheye:"
        if data.get("omni:lensdistortion:model") == "opencvFisheye":
            w0, h0 = ast.literal_eval(data[prefix + "imageSize"])
            fx, fy, cx, cy = [float(data[prefix + x]) for x in ("fx", "fy", "cx", "cy")]
            h, w = hw if hw else (h0, w0)
            sx, sy = w / w0, h / h0
            k = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], device=self.device).expand(self.num_envs, -1, -1).contiguous()
            d = torch.tensor([float(data[prefix + f"k{i}"]) for i in range(1, 5)], device=self.device).expand(self.num_envs, -1).contiguous()
            camera = FisheyeCameraBatch(env.camera.world_to_camera.clone(), k, w, h, distortion=d, image_circle_radius=w / 2)
        else:
            camera = CameraBatch.from_imported_pinhole(env.camera.world_to_camera.clone(), data)
        return {"camera": camera, "cid": cid, "frame": None}

    @torch.no_grad()
    def render_all(self) -> dict[str, torch.Tensor]:
        """uint8 [N, H, W, 3] per camera name (GPU)."""
        env = self.env
        env.render_enabled[:] = True
        out = {}
        for name, rig in self.rigs.items():
            env.camera, env.camera_name, env.last_frame = rig["camera"], rig["cid"], rig["frame"]
            frame = env.render()
            rig["frame"] = frame
            out[name] = (frame.rgb.clamp(0, 1) * 255).round().to(torch.uint8)
        return out

    # ------------------------------------------------------------------ state
    @torch.no_grad()
    def ee_poses(self):
        """TCP poses in the robot base frame: (R [N,2,3,3], p [N,2,3]) right-then-left, plus world base (Rb, pb)."""
        p = self.p
        Rb = p.body_rot[:, self.base_body].reshape(-1, 3, 3)
        pb = p.body_pos[:, self.base_body]
        Rw = p.body_rot[:, self.tip_ids].reshape(-1, 2, 3, 3)
        pw = p.body_pos[:, self.tip_ids]
        R = Rb.transpose(-1, -2)[:, None] @ Rw
        pos = torch.einsum("nji,nkj->nki", Rb, pw - pb[:, None])
        return R, pos, Rb, pb

    @torch.no_grad()
    def state(self) -> torch.Tensor:
        R, pos, _, _ = self.ee_poses()
        grip = (self.p.qpos[:, self.grip_qadr] / GRIP_CLOSED_CTRL).clamp(0, 1)  # [N,2] right,left
        rv = mat_to_rotvec(R)
        parts = []
        for k in range(2):
            parts += [grip[:, k : k + 1], pos[:, k], rv[:, k]]
        return torch.cat(parts, -1).float()  # [N,14]

    def observe(self) -> dict:
        obs = {"states": self.state()}
        if self.rigs:
            obs.update(self.render_all())
        return obs

    # ------------------------------------------------------------------ reset
    def _sample_objects(self, qpos_row: torch.Tensor, rng: np.random.Generator):
        """Port of PickPlaceTask.spawn's object placement (layout center: slots alternate sides)."""
        from scipy.spatial.transform import Rotation

        placed = []
        bodies = []
        for i, slot in enumerate(self.slots):
            variants = slot.get("variants", [slot])
            chosen = variants[int(rng.integers(len(variants)))]
            side = "left" if i % 2 == 0 else "right"
            dims = np.array(self._dims[chosen["asset"]]); hx, hy, hz = dims / 2
            corners = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0, dims[2])])
            for _ in range(200):
                yaw = rng.uniform(-np.pi, np.pi)
                resting = np.eye(3) if chosen["asset"] == "RTC-001" else Rotation.from_euler(
                    "xyz", [(0, 0, 0), (90, 0, 0), (-90, 0, 0), (0, 90, 0), (0, -90, 0), (180, 0, 0)][int(rng.integers(6))], degrees=True).as_matrix()
                rot = Rotation.from_euler("z", yaw).as_matrix() @ resting
                t = corners @ rot.T; lo, hi = t.min(0), t.max(0); half = (hi - lo) / 2
                xmin, xmax = (1.58, 1.79) if side == "left" else (2.43, 2.64)
                x, y = rng.uniform(xmin, xmax), rng.uniform(0.95, 1.10)
                if x - half[0] < 1.34 or x + half[0] > 2.78 or y - half[1] < 0.875 or y + half[1] > 1.26: continue
                if 1.84 - half[0] < x < 2.38 + half[0]: continue
                if any(abs(x - a) < half[0] + ha + 0.025 and abs(y - b) < half[1] + hb + 0.025 for a, b, ha, hb in placed): continue
                placed.append((x, y, half[0], half[1])); break
            else:
                raise RuntimeError("no collision-free spawn footprint; reduce spawn_count")
            pos = np.array([x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, self._shelf_top - lo[2] + 0.008])
            quat = np.roll(Rotation.from_matrix(rot).as_quat(), 1)
            adr = self.model.joint(chosen["joint"]).qposadr[0]
            qpos_row[adr : adr + 7] = torch.tensor(np.r_[pos, quat], device=self.device, dtype=torch.float32)
            bodies.append(self.model.body(chosen["body"]).id)
        return bodies

    @torch.no_grad()
    def reset(self, mask: torch.Tensor | None = None, seeds: list[int] | None = None) -> dict:
        from scipy.spatial.transform import Rotation

        N = self.num_envs
        if mask is None:
            mask = torch.ones(N, device=self.device, dtype=torch.bool)
        qpos = self.p.default_qpos.clone()  # the published ready keyframe (arms) + parked variants
        mocap_pos, mocap_quat = self.p.mocap_pos.clone(), self.p.mocap_quat.clone()
        mocap_id = self.model.body_mocapid[self.base_body]
        base_quat = torch.tensor(np.roll(Rotation.from_euler("z", 90.0, degrees=True).as_quat(), 1), device=self.device, dtype=torch.float32)
        if not hasattr(self, "_active_bodies"):
            self._active_bodies = [list(self.object_bodies) for _ in range(N)]
        for i in torch.nonzero(mask, as_tuple=True)[0].tolist():
            rng = np.random.default_rng(seeds[i] if seeds is not None else self.rng.integers(2**31))
            self._active_bodies[i] = self._sample_objects(qpos[i], rng)
            mocap_pos[i, mocap_id] = torch.tensor(self._base, device=self.device, dtype=torch.float32)
            mocap_quat[i, mocap_id] = base_quat
        self.env.reset(mask, qpos=qpos, mocap_position=mocap_pos, mocap_quaternion=mocap_quat)
        # settle: position servos hold the keyframe (reset copies qpos into ctrl)
        n = int(round(self.cfg.settle_s * self.env.config.physics_hz))
        if n:
            self.p.step(self.p.ctrl.clone(), n)
        self.elapsed = torch.where(mask, torch.zeros_like(self.elapsed), self.elapsed)
        n_obj = len(self.slots)
        if not hasattr(self, "_obj_init_pos"):
            self._obj_init_pos = torch.zeros(N, n_obj, 3, device=self.device)
            self._was_in = torch.zeros(N, n_obj, dtype=torch.bool, device=self.device)
            self._regress_paid = torch.zeros(N, n_obj, dtype=torch.bool, device=self.device)
        pos = self.object_positions()
        self._obj_init_pos[mask] = pos[mask]
        self._was_in[mask] = False
        self._regress_paid[mask] = False
        for rig in self.rigs.values():
            if rig["frame"] is not None:
                rig["frame"].valid[mask] = False
        return self.observe()

    # ------------------------------------------------------------------ control
    def _ik(self, q_world_targets_R: torch.Tensor, p_world_targets: torch.Tensor) -> np.ndarray:
        """Damped least squares on MuJoCo Jacobians, per env on CPU. Returns arm joint targets [N,14]."""
        m, d = self.model, self._check
        qpos_all = self.p.qpos.cpu().numpy()
        mocap_pos, mocap_quat = self.p.mocap_pos.cpu().numpy(), self.p.mocap_quat.cpu().numpy()
        Rt, pt = q_world_targets_R.cpu().numpy(), p_world_targets.cpu().numpy()
        out = np.zeros((self.num_envs, 14), dtype=np.float32)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lam2 = self.cfg.ik_damping**2
        for i in range(self.num_envs):
            d.qpos[:] = qpos_all[i]; d.mocap_pos[:] = mocap_pos[i]; d.mocap_quat[:] = mocap_quat[i]
            for _ in range(self.cfg.ik_iters):
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                for k, tip in enumerate(self.tip_ids):  # 0 right, 1 left
                    cols = self.arm_dofadr[7 * k : 7 * k + 7]
                    mujoco.mj_jacBody(m, d, jacp, jacr, tip)
                    J = np.vstack([jacp[:, cols], jacr[:, cols]])  # 6x7
                    R_cur = d.xmat[tip].reshape(3, 3); p_cur = d.xpos[tip]
                    e_rot = _rotvec_np(Rt[i, k] @ R_cur.T)
                    e = np.r_[pt[i, k] - p_cur, e_rot]
                    dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
                    qa = self.arm_qadr[7 * k : 7 * k + 7]
                    d.qpos[qa] = np.clip(d.qpos[qa] + dq, self.arm_range[7 * k : 7 * k + 7, 0], self.arm_range[7 * k : 7 * k + 7, 1])
            out[i] = d.qpos[self.arm_qadr]
        return out

    @torch.no_grad()
    def step(self, action: torch.Tensor | np.ndarray):
        """action [N,14]: grip_R, dxyz_R, rotvec_R, grip_L, dxyz_L, rotvec_L (EE-local per-step deltas)."""
        t0 = time.time()
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.num_envs, 14)
        R, pos, Rb, pb = self.ee_poses()
        dpos = torch.stack([a[:, 1:4], a[:, 8:11]], 1)
        drot = torch.stack([a[:, 4:7], a[:, 11:14]], 1)
        grip = torch.stack([a[:, 0], a[:, 7]], 1).clamp(0, 1)
        # integrate in the base frame, then express in world for the IK
        pos_t = pos + torch.einsum("nkij,nkj->nki", R, dpos)
        R_t = R @ rotvec_to_mat(drot)
        Rw_t = Rb[:, None] @ R_t
        pw_t = pb[:, None] + torch.einsum("nij,nkj->nki", Rb, pos_t)
        t1 = time.time()
        q = self._ik(Rw_t, pw_t)
        t2 = time.time()
        ctrl = self.p.ctrl.clone()
        ctrl[:, self.arm_act] = torch.from_numpy(q).to(self.device)
        if self.cfg.gripper_mode == "binary":
            grip = (grip > self.cfg.gripper_thresh).float()
        elif self.cfg.gripper_mode == "boost":
            grip = (grip / self.cfg.gripper_thresh).clamp(0, 1)
        ctrl[:, self.grip_act] = GRIP_CLOSED_CTRL * grip
        self.p.step(ctrl, self.frame_steps)
        t3 = time.time()
        self.elapsed += 1
        obs = self.observe()
        t4 = time.time()
        reward, success = self.compute_reward()
        fell = self.fallen()
        if self.cfg.reward.fail_on_fall:
            reward = reward - self.cfg.reward.fall_penalty * fell.float()
        terminated = success | (fell if self.cfg.reward.fail_on_fall else torch.zeros_like(fell))
        truncated = self.elapsed >= self.cfg.max_episode_steps
        self.timing = {"integrate": t1 - t0, "ik": t2 - t1, "physics": t3 - t2, "observe": t4 - t3}
        return obs, reward, terminated, truncated, {"success": success, "fail": fell, "reward_terms": self.last_reward_terms}

    # ------------------------------------------------------------------ reward
    @torch.no_grad()
    def grasp_contacts(self) -> torch.Tensor:
        """[N, 2, n_obj] bool: finger geoms of arm k touch object j (either contact geom order)."""
        c = self.p.contacts()
        g = c["geom"].long()  # [N, cap, 2]
        valid = c["valid"]  # [N, cap]
        n = len(self.slots)
        out = torch.zeros(self.num_envs, 2 * n + 1, dtype=torch.bool, device=self.device)
        for a_col, o_col in ((0, 1), (1, 0)):
            arm = self.arm_of_geom[g[..., a_col]]
            obj = self.obj_of_geom[g[..., o_col]]
            ok = valid & (arm >= 0) & (obj >= 0)
            idx = torch.where(ok, arm * n + obj, torch.full_like(arm, 2 * n))
            out.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
        return out[:, : 2 * n].reshape(self.num_envs, 2, n)

    @torch.no_grad()
    def compute_reward(self):
        """Reward per env and the success flag; fills ``self.last_reward_terms`` for logging."""
        R = self.cfg.reward
        N, n = self.num_envs, len(self.slots)
        o = self.object_positions()  # [N, n, 3]
        inside = self.in_basket(o)  # [N, n]
        success = inside.all(-1)
        terms = {"n_in": inside.sum(-1).float()}
        if not R.staged:
            reward = R.success * success.float()
            if R.reach_weight > 0:
                reward = reward - R.reach_weight * self.min_tcp_object_distance()
            self.last_reward_terms = terms
            return reward, success
        phi = lambda d: 1.0 - torch.tanh(R.beta * d)
        tcp = self.p.body_pos[:, self.tip_ids]  # [N, 2, 3]
        d = torch.cdist(tcp, o)  # [N, 2, n]
        closed = (self.p.qpos[:, self.grip_qadr] / GRIP_CLOSED_CTRL).clamp(0, 1)  # [N, 2]
        grasp = self.grasp_contacts() & (closed > R.grasp_close_thresh)[:, :, None]  # [N, 2, n]
        # active object: closest (over arms) among those not yet in the basket
        d_min_arm = d.amin(1)  # [N, n]
        cand = torch.where(inside, torch.full_like(d_min_arm, 1e6), d_min_arm)
        active = cand.argmin(-1)  # [N]
        ar = torch.arange(N, device=self.device)
        d_act = d[ar, :, active]  # [N, 2]
        acting = d_act.argmin(-1)  # [N]
        idle = 1 - acting
        o_act = o[ar, active]  # [N, 3]
        g_act = grasp[ar, :, active].any(-1)  # [N]
        lifted = g_act & ((o_act[:, 2] - self._obj_init_pos[ar, active, 2]) >= R.lift_height)
        in_act = inside[ar, active]
        d_goal = torch.linalg.vector_norm(o_act - self.basket_target, dim=-1)
        u_act = 1.0 - closed[ar, acting]
        stage = 2.0 * phi(d_act.amin(-1))
        stage = torch.where(g_act, 4.0 + phi(d_goal), stage)
        stage = torch.where(lifted, 6.0 + phi(d_goal), stage)
        stage = torch.where(in_act, 8.0 + R.release_bonus * u_act, stage)
        stage = stage + R.completion_bonus * inside.sum(-1).float()
        stage = torch.where(success, torch.full_like(stage, R.success + R.completion_bonus * (n - 1)), stage)
        reward = stage
        terms.update({"stage": stage, "grasp": g_act.float(), "lifted": lifted.float(), "d_reach": d_act.amin(-1), "d_goal": d_goal})
        if R.idle_arm_penalty > 0:
            q = self.p.qpos[:, torch.from_numpy(self.arm_qadr).to(self.device)].reshape(N, 2, 7)
            dev = ((q - self.q_home[None]) ** 2).sum(-1)  # [N, 2]
            pen = R.idle_arm_penalty * dev[ar, idle]
            reward = reward - pen
            terms["idle_pen"] = pen
        if R.disturb_penalty > 0:
            moved = torch.tanh(R.beta * torch.linalg.vector_norm(o - self._obj_init_pos, dim=-1))  # [N, n]
            mask = ~inside
            mask[ar, active] = False
            pen = R.disturb_penalty * (moved * mask.float()).sum(-1)
            reward = reward - pen
            terms["disturb_pen"] = pen
        if R.regress_penalty > 0:
            left = self._was_in & ~inside & ~self._regress_paid
            pen = R.regress_penalty * left.any(-1).float()
            self._regress_paid |= left
            reward = reward - pen
            terms["regress_pen"] = pen
        self._was_in |= inside
        self.last_reward_terms = terms
        return reward, success

    @torch.no_grad()
    def fallen(self) -> torch.Tensor:
        """Any object below its reset height by more than fall_margin and not in the basket (knocked off the shelf)."""
        o = self.object_positions()
        if not hasattr(self, "_obj_init_pos"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dropped = (self._obj_init_pos[..., 2] - o[..., 2]) > self.cfg.reward.fall_margin
        return (dropped & ~self.in_basket(o)).any(-1)

    @torch.no_grad()
    def in_basket(self, o: torch.Tensor) -> torch.Tensor:
        return (o[..., 0] > 1.91) & (o[..., 0] < 2.31) & (o[..., 1] > 0.92) & (o[..., 1] < 1.18) \
            & (o[..., 2] > self._shelf_top - 0.02) & (o[..., 2] < self._basket_rim + 0.08)

    @torch.no_grad()
    def object_positions(self) -> torch.Tensor:
        idx = torch.tensor(self._active_bodies, device=self.device)  # [N, n_obj]
        return torch.gather(self.p.body_pos, 1, idx[..., None].expand(-1, -1, 3))

    @torch.no_grad()
    def success(self) -> torch.Tensor:
        """Every object inside the basket AABB used by the data generator (world frame)."""
        return self.in_basket(self.object_positions()).all(-1)

    @torch.no_grad()
    def min_tcp_object_distance(self) -> torch.Tensor:
        o = self.object_positions()  # [N,n,3]
        tcp = self.p.body_pos[:, self.tip_ids]  # [N,2,3]
        d = torch.cdist(tcp, o)  # [N,2,n]
        return d.amin(dim=(1, 2))


def _rotvec_np(R: np.ndarray) -> np.ndarray:

    return Rotation.from_matrix(R).as_rotvec()
