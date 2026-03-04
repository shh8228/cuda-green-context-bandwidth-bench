#!/usr/bin/env bash
# run_all.sh — Auto-detect GPU, build, run, and save results with GPU name suffix.
# Usage: bash run_all.sh [buffer_MB] [iterations] [gpu_id] [sm_step] [trials]
#
# Works on: A100 (sm_80), H100/H200 (sm_90a), Blackwell (sm_100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUF_MB="${1:-512}"
ITERS="${2:-20}"
GPU_ID="${3:-0}"
SM_STEP="${4:-0}"
TRIALS="${5:-5}"

# --- Auto-detect GPU ---
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" | head -1 | xargs)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU_ID" | head -1 | xargs)

# Map compute capability to nvcc arch flag
CC_MAJOR="${COMPUTE_CAP%%.*}"
CC_MINOR="${COMPUTE_CAP##*.}"

case "${CC_MAJOR}.${CC_MINOR}" in
    10.0) ARCH="sm_100" ;;
    9.0)  ARCH="sm_90a" ;;
    8.9)  ARCH="sm_89"  ;;
    8.6)  ARCH="sm_86"  ;;
    8.0)  ARCH="sm_80"  ;;
    7.5)  ARCH="sm_75"  ;;
    7.0)  ARCH="sm_70"  ;;
    *)    ARCH="sm_${CC_MAJOR}${CC_MINOR}"
          echo "WARNING: Untested architecture ${COMPUTE_CAP}, trying ${ARCH}" ;;
esac

# Create a filesystem-safe GPU tag (e.g., "H100_SXM" or "A100_80GB")
GPU_TAG=$(echo "$GPU_NAME" | sed 's/NVIDIA //; s/ /_/g; s/[^A-Za-z0-9_-]//g')

echo "============================================"
echo "  GPU:           $GPU_NAME"
echo "  Compute Cap:   $COMPUTE_CAP"
echo "  NVCC Arch:     $ARCH"
echo "  Output Tag:    $GPU_TAG"
echo "  Buffer:        ${BUF_MB} MB"
echo "  Iterations:    ${ITERS}"
echo "  Trials:        ${TRIALS}"
echo "============================================"
echo ""

# --- Build ---
echo "=== Building for ${ARCH} ==="
make clean
make CUDA_ARCH="$ARCH"
echo ""

# --- Run ---
echo "=== Running benchmark ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" ./green_ctx_bw_bench "$BUF_MB" "$ITERS" 0 "$SM_STEP" "$TRIALS" \
    > "results_${GPU_TAG}.csv" \
    2> "results_${GPU_TAG}.log"

echo "Benchmark complete. Generating plot..."

# --- Plot ---
python3 plot_results.py "results_${GPU_TAG}.csv" "bw_saturation_${GPU_TAG}.png"

echo ""
echo "=== Saved ==="
echo "  CSV:  results_${GPU_TAG}.csv"
echo "  Log:  results_${GPU_TAG}.log"
echo "  Plot: bw_saturation_${GPU_TAG}.png"

# --- Print log (stderr output from benchmark) ---
echo ""
echo "=== Benchmark Log ==="
cat "results_${GPU_TAG}.log"
