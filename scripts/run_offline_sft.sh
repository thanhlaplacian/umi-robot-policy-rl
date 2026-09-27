#!/usr/bin/env bash
# usage: bash scripts/run_offline_sft.sh <config under configs/offline, without .yaml> [hydra overrides...]
# Requires a running Ray head (see docs/PHASE1.md): ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$1"; shift || true
source "${VENV:-/home/thanh/venvs/umi-rl}/bin/activate"
export EMBODIED_PATH="$ROOT/third_party/rlinf/examples/sft"   # hydra searchpath for RLinf's hybrid_engines/* defaults
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export TOKENIZERS_PARALLELISM=false
export STEAM_AUTOSCORE=0 FLARE_EMBED=0
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
LOG_DIR="${LOG_DIR:-/home/thanh/rlinf-runs/logs/umi-rl}"
mkdir -p "$LOG_DIR"
cd "$ROOT"
python scripts/train_offline_sft.py --config-name "offline/$CFG" runner.logger.log_path="$LOG_DIR" "$@"
