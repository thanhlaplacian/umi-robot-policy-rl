#!/usr/bin/env bash
# Stop any offline SFT driver started by run_offline_sft.sh. The pattern is assembled at runtime so
# a caller whose own command line mentions the script name is not matched (pkill -f would kill it).
pat="train_offline""_sft.py"
pkill -f "$pat" || true
