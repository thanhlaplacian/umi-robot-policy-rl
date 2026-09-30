#!/usr/bin/env bash
# Kill processes whose command line matches the CONCATENATION of the arguments, e.g.
#   bash scripts/kill_procs.sh train_rl .py        # matches "train_rl.py"
# Split the pattern so the caller's own shell command line never contains the full match
# (pkill -f from an interactive/agent shell otherwise kills that shell: exit code 144).
set -u
pat="$(printf '%s' "$@")"
pkill -f -- "$pat" && echo "killed: $pat" || echo "no process matched: $pat"
