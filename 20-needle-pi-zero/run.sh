#!/usr/bin/env bash
# The full run from the video, on a Raspberry Pi Zero 2 W. Usage: ./run.sh <weights.cact> <label>
#   ./run.sh needle3.cact needle3          # about 30 min
#   ./run.sh needle3-4l.cact needle3-4l    # about 10 min
# Needle 2 runs the same way from a venv with cactus-needle 2.0.12 instead of cactus-compute.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src" CACTUS_TELEMETRY=0 NEEDLE_TELEMETRY=0
NEEDLE_WEIGHTS="$1" python scripts/run_benchmark.py --tool-counts 5 10 25 50 --repetitions 5 --warmup 3 \
    --seed 42 --max-new-tokens 256 --label "$2"
