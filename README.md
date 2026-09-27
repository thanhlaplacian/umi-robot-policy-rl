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
