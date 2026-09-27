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
