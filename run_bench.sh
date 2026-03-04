#!/usr/bin/env bash
# run_bench.sh — Build and run Green Context bandwidth saturation benchmark
# Usage: bash run_bench.sh [buffer_MB] [iterations] [gpu_id] [sm_step]
#
# Container: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Target:    NVIDIA Blackwell (GB200)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUF_MB="${1:-512}"
ITERS="${2:-20}"
GPU_ID="${3:-0}"
SM_STEP="${4:-0}"

echo "=== Building benchmark ==="
make clean
make all

echo ""
echo "=== Running benchmark ==="
echo "  Buffer: ${BUF_MB} MB, Iterations: ${ITERS}, GPU: ${GPU_ID}, SM step: ${SM_STEP}"
echo ""

./green_ctx_bw_bench "$BUF_MB" "$ITERS" "$GPU_ID" "$SM_STEP" | tee results.csv

echo ""
echo "=== Generating plot ==="
python3 plot_results.py results.csv

echo ""
echo "=== Done ==="
echo "Results: results.csv"
echo "Plot:    bw_saturation.png"
