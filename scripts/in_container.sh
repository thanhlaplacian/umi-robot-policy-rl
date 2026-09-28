#!/usr/bin/env bash
# Run a command inside the SFT container from this repo's root with the umi-rl venv on PATH.
#   bash scripts/in_container.sh [-g GPUS] <command...>
# e.g. bash scripts/in_container.sh -g 7 python scripts/gym_env_smoke.py --num-envs 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${CONTAINER:-umi-thanh-maskfix}"
VENV="${VENV:-/home/thanh/venvs/umi-rl}"
GPUS="${CUDA_VISIBLE_DEVICES:-}"
if [ "${1:-}" = "-g" ]; then GPUS="$2"; shift 2; fi
printf -v CMD '%q ' "$@"
docker exec -e CUDA_VISIBLE_DEVICES="$GPUS" "$CONTAINER" bash -lc \
  "export PATH=$VENV/bin:/usr/local/cuda/bin:\$PATH HF_HUB_CACHE=/home/thanh/.cache/huggingface/hub STEAM_AUTOSCORE=0 FLARE_EMBED=0 TOKENIZERS_PARALLELISM=false; cd $ROOT && $CMD"
