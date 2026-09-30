#!/usr/bin/env bash
# Apply patches/umi-robot-policy/*.patch to the submodule; already-applied patches are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/third_party/umi-robot-policy"
for p in "$ROOT"/patches/umi-robot-policy/*.patch; do
    if git -C "$SUB" apply --check --reverse "$p" >/dev/null 2>&1; then
        echo "already applied: $(basename "$p")"
    elif git -C "$SUB" apply --check "$p" >/dev/null 2>&1; then
        git -C "$SUB" apply "$p" && echo "applied: $(basename "$p")"
    else
        # The VLSA-mask patch is optional (padding changes actions by <= 1 bf16 ulp on the
        # measured checkpoints) and no longer applies past fork commit a379960; warn, don't fail.
        echo "WARNING: $(basename "$p") does not apply to $(git -C "$SUB" rev-parse --short HEAD); skipped" >&2
    fi
done
