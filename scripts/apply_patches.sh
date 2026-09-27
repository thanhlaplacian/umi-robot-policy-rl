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
        echo "ERROR: $(basename "$p") does not apply to $(git -C "$SUB" rev-parse --short HEAD)" >&2; exit 1
    fi
done
