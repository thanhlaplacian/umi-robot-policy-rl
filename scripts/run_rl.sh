#!/usr/bin/env bash
# usage: bash scripts/run_rl.sh <config under configs/rl, without .yaml> [hydra overrides...]
# Needs a Ray head with the dashboard (see docs/PHASE1.md). Meant to run inside the SFT container.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$1"; shift || true
source "${VENV:-/home/thanh/venvs/umi-rl}/bin/activate"
export UMI_RL_ROOT="$ROOT"
export EMBODIED_PATH="$ROOT/third_party/rlinf/examples/embodiment"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export TOKENIZERS_PARALLELISM=false STEAM_AUTOSCORE=0 FLARE_EMBED=0
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
LOG_DIR="${LOG_DIR:-/home/thanh/rlinf-runs/logs/umi-rl}"
mkdir -p "$LOG_DIR"
cd "$ROOT"
python scripts/train_rl.py --config-name "$CFG" runner.logger.log_path="$LOG_DIR" "$@"
