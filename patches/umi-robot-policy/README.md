# Patches applied to the `umi-robot-policy` submodule

Applied by `scripts/apply_patches.sh` (idempotent: skips a patch that is already in). They are
env-gated so the submodule behaves as upstream unless the variable is set.

| Patch | Gate | Why |
|---|---|---|
| `0001-vlsa-attention-mask-and-padding-side.patch` | `GR00T_VLSA_APPLY_MASK=1`, `GR00T_TEXT_PADDING_SIDE=left|right` | `gr00t_n1d7` `vl_self_attention` runs without the prompt attention mask, so padded prompt tokens change the predicted action (measured on LIBERO: 5/22/90/414 pad tokens → mean |Δaction| 0.17/0.27/0.88/1.25 at |a|≈0.22). RLinf batches prompts of different length with padding during rollout/training, so the mask is required for padding-invariant actions. SFT A/B (30k steps, 2026-09-25) showed no measurable difference in holdout ADE with or without the fix, so it is inference-safety only. |
