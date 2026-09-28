#!/usr/bin/env bash
# Install the laplacian-gym CORE (physics + rendering, no planners/viewer/data extras) into the
# umi-rl venv. Run inside the SFT container (nvcc 13.x, gcc) so builds do not touch the host:
#   docker exec umi-thanh-maskfix bash -lc 'cd <repo> && bash scripts/setup_gym.sh'
# Verified 2026-09-28 with torch 2.11.0+cu130: the gym's torch==2.14 pin is a measured stack,
# not an API requirement (pyproject says torch>=2.7; bootstrap only checks CUDA 13.0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/home/thanh/venvs/umi-rl}"
PY="$VENV/bin/python"
UV="${UV:-/home/thanh/.local/bin/uv}"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GYM="$ROOT/third_party/laplacian-gym"

git -C "$ROOT" submodule update --init third_party/laplacian-gym
# LFS assets (3.1 GB): pulled from GitHub unless already materialized
if [ ! -s "$GYM/assets/scenes/livinglab_hq/point_cloud.ply" ] || [ "$(stat -c %s "$GYM/assets/scenes/livinglab_hq/point_cloud.ply")" -lt 1000000 ]; then
    git -C "$GYM" lfs pull
fi
# core deps; numpy>=2 is required by the gym (only rlinf-libero, unused here, wants numpy<2)
"$UV" pip install --python "$PY" "mujoco==3.13.0" "mujoco-warp==3.13.0" "warp-lang==1.17.0" "gsplat==1.5.3" \
    "plyfile>=1.1" "PyYAML>=6" "scipy>=1.14" "trimesh>=4.6" "Pillow>=11" ninja
"$UV" pip install --python "$PY" -e "$GYM" --no-deps
"$UV" pip install --python "$PY" --no-cache --no-build-isolation \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
# Warp 1.17.0 + mujoco-warp 3.13.0: CCD kernel meta key missing -> the gym's idempotent patch
(cd "$GYM" && "$PY" scripts/patch_mujoco_warp.py)
"$PY" -c "import torch; assert torch.__version__.startswith('2.11'), torch.__version__; print('torch', torch.__version__)"
(cd "$GYM" && "$PY" -m laplacian_gym.cli doctor | head -20)
echo "gym core installed into $VENV; first run compiles warp/gsplat/nvdiffrast kernels (~10 min)"
