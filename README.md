# umi-robot-policy-rl

Reinforcement-learning fine-tuning of the company GR00T N1.7 policy (`umi-robot-policy`) with
[RLinf](https://github.com/RLinf/RLinf).

Layout

| Path | What | How it is tracked |
|---|---|---|
| `third_party/rlinf` | RLinf framework | git subtree (upstream `ee2cab6`), local fixes committed on top |
| `third_party/umi-robot-policy` | company policy fork (model, processor, data SDK) | git submodule, pinned, never edited here |
| `patches/umi-robot-policy/` | optional patches applied to the submodule at setup time | plain diff files |
| `umi_rl/` | glue code: embodiment tag, obs/action converters, model config, tools | python package |
| `configs/` | Hydra configs for RLinf runs (model, offline RL, PPO) | yaml |
| `scripts/` | environment setup, parity check, launchers | bash / python |
| `docs/` | plan and phase reports | markdown |

See `docs/PLAN.md` for the phased plan and `docs/PHASE0.md` for the phase-0 report.

## Quick start

```bash
# one-time: venv (RLinf installer, fork as GR00T_PATH), ~40 min
bash scripts/setup_env.sh
source /home/thanh/venvs/umi-rl/bin/activate

# phase 0 checks (single GPU each)
python scripts/parity_check.py --dataset /data/dataset/26-W33-TELE2-rebot-batch1   # fork vs RLinf actions, expect 0.0
python scripts/rl_path_smoke.py                                                   # train-mode rollout + actor forward/backward
python scripts/padding_invariance.py                                              # prompt padding sensitivity

# phase 1: offline training on 8 GPUs through RLinf's FSDP SFT stack
ray start --head --port=6379 --num-cpus=64 --dashboard-host=127.0.0.1 --dashboard-port=8265
bash scripts/run_offline_sft.sh umi_sft_smoke                                     # configs/offline/umi_sft_smoke.yaml
bash scripts/kill_offline_sft.sh                                                  # stop a run
python scripts/export_hf_checkpoint.py --step-dir <log>/checkpoints/global_step_N \
    --base /data/models/UMICore-v0.16.9/checkpoint-3000 --out /home/thanh/models/<name>   # fork-loadable checkpoint
python scripts/steam_label_audit.py                                               # STEAM label coverage of a mix
```

Status: phase 0 complete (`docs/PHASE0.md`), phase 1 increment 1 complete (`docs/PHASE1.md`).

## Quick start: RL training in the simulator (LaplacianRLEnv)

Everything below runs inside the SFT container (`umi-thanh-maskfix`) so builds never touch the
shared host; `scripts/in_container.sh` runs a command there from the repo root with the venv on
PATH. Details and measurements: `docs/PHASE2.md`.

```bash
# one-time: laplacian-gym core into the umi-rl venv (submodule + LFS assets, mujoco-warp patch,
# gsplat/nvdiffrast builds). First simulator start then compiles kernels for ~10 min.
bash scripts/in_container.sh bash scripts/setup_gym.sh

# 1. simulator alone: 100 physics/render steps, 2 envs
bash scripts/in_container.sh -g 7 bash -c 'cd third_party/laplacian-gym && python -m laplacian_gym.cli run --config configs/livinglab_hq.yaml --steps 100 --num-envs 2'

# 2. env + policy closed-loop smoke: reset, hold, scripted move, 3 policy chunks; saves camera frames
bash scripts/in_container.sh -g 7 python scripts/gym_env_smoke.py --num-envs 2 --chunks 3
#    frames -> /home/thanh/rlinf-runs/logs/umi-rl/gym_smoke/*.png

# 3. PPO through RLinf (env 0-1, rollout 2-3, actor 4-7), 2 epochs
bash scripts/in_container.sh bash -c 'ray start --head --port=6380 --num-cpus=48 --dashboard-host=127.0.0.1 --dashboard-port=8266'
bash scripts/in_container.sh bash -c 'export RAY_ADDRESS=127.0.0.1:6380; bash scripts/run_rl.sh laplacian_ppo_gr00t_umi_smoke'
#    logs/tensorboard: /home/thanh/rlinf-runs/logs/umi-rl ; W&B project finetune-gr00t-n1d7
```

To train for real, copy `configs/rl/laplacian_ppo_gr00t_umi_smoke.yaml`, raise `runner.max_epochs`,
`env.train.total_num_envs` (must be divisible by the number of env GPUs) and
`max_steps_per_rollout_epoch` (multiple of 16), and set `save_interval`. Env knobs live in
`configs/env/laplacian_pick_place.yaml` (`init_params`: `spawn_count`, cameras, `wrist_hw`,
`dense_reward`, `prompt`). Stop a run with `pkill -f train_rl` inside the container; a run's
checkpoints export to a fork-loadable directory with `scripts/export_hf_checkpoint.py`.
