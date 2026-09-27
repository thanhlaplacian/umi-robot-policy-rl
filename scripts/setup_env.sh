#!/usr/bin/env bash
# Build the virtualenv for umi-robot-policy-rl.
#
#   VENV=/home/thanh/venvs/umi-rl DOWNLOAD_DIR=/home/thanh/.rlinf-assets bash scripts/setup_env.sh
#
# Uses RLinf's own installer (embodied stack, gr00t_n1d7 model, maniskill_libero env for the sim
# smoke tests) with GR00T_PATH pointing at the pinned umi-robot-policy submodule, so the venv
# imports the company fork's `gr00t` package instead of upstream Isaac-GR00T. Then installs the
# data SDK, RLinf and this package as editable, without touching pinned deps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/home/thanh/venvs/umi-rl}"
export DOWNLOAD_DIR="${DOWNLOAD_DIR:-/home/thanh/.rlinf-assets}"
export GR00T_PATH="$ROOT/third_party/umi-robot-policy"

git -C "$ROOT" submodule update --init third_party/umi-robot-policy
git -C "$ROOT/third_party/umi-robot-policy" submodule update --init deps/umi-data-sdk
bash "$ROOT/scripts/apply_patches.sh"

cd "$ROOT/third_party/rlinf"
# --no-root: system packages (cmake, libgl, ...) are already present on this host; sys_deps.sh needs sudo.
bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero --venv "$VENV" --no-root

# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -e "$ROOT/third_party/umi-robot-policy/deps/umi-data-sdk" --no-deps
uv pip install -e "$ROOT/third_party/rlinf" --no-deps
uv pip install -e "$ROOT" --no-deps
ln -sfn "$VENV" "$ROOT/.venv"
python - <<'PY'
import gr00t, rlinf, umi_data_sdk, torch
print("gr00t from", gr00t.__file__)
print("rlinf from", rlinf.__file__)
print("umi_data_sdk from", umi_data_sdk.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "venv ready: $VENV"
