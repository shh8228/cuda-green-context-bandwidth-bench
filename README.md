# Green Context DRAM Bandwidth Saturation Benchmark

Microbenchmark to empirically determine the **minimum number of SMs required to saturate GPU DRAM read bandwidth** using CUDA Green Contexts, targeting NVIDIA Blackwell (GB200).

## Motivation

When spatially partitioning a GPU with CUDA Green Contexts (e.g., for disaggregated serving, MPS-like workload isolation, or co-located inference), a key question is: **how many SMs are actually needed to saturate memory bandwidth?** If a memory-bound kernel saturates DRAM bandwidth with only a fraction of available SMs, the remaining SMs can be allocated to compute-bound work without degrading memory throughput.

## How It Works

1. **Baseline**: Measures full-GPU DRAM read bandwidth using all SMs.
2. **Sweep**: For each SM count from the architecture minimum up to the total:
   - Creates a **Green Context** with that many SMs via `cuDevSmResourceSplitByCount` + `cuGreenCtxCreate`.
   - Launches a pure streaming-read kernel (`__ldg` float4 loads, coalesced access pattern) confined to the Green Context.
   - Measures achieved read bandwidth using CUDA events, taking the **median of multiple trials** for robustness.
3. **Analysis**: Identifies the saturation point — the smallest SM count achieving ≥95% of full-GPU bandwidth.

### SM Granularity by Architecture

| Architecture | Compute Capability | Min SMs | Alignment |
|---|---|---|---|
| Pascal | 6.x | 1 | 1 |
| Volta / Turing | 7.x | 2 | 2 |
| Ampere | 8.x | 4 | 2 |
| Hopper | 9.x | 8 | 8 |
| **Blackwell** | **10.x** | **8** | **8** |

The benchmark auto-detects the architecture and uses the correct granularity.

## Prerequisites

- **Container**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (or any environment with CUDA ≥ 12.4 and `nvcc`)
- **GPU**: NVIDIA Blackwell GB200 (also works on Hopper, Ampere, etc.)
- **Python 3** with `matplotlib` (for plotting; analysis text output works without it)

## Build

```bash
# Default: Blackwell (sm_100)
make

# For other architectures:
make CUDA_ARCH=sm_90a   # Hopper
make CUDA_ARCH=sm_86    # Ampere (A6000, A100, etc.)
```

## Run

### One-command (build + run + plot)

```bash
bash run_bench.sh [buffer_MB] [iterations] [gpu_id] [sm_step]

# Example:
bash run_bench.sh 512 20 0 0
```

### Manual

```bash
./green_ctx_bw_bench [buffer_MB] [iterations] [gpu_id] [sm_step] [trials]
```

| Argument | Default | Description |
|---|---|---|
| `buffer_MB` | 512 | Size of the read buffer in MB. Larger = more stable results. |
| `iterations` | 20 | Kernel launches per timed trial. |
| `gpu_id` | 0 | CUDA device index. |
| `sm_step` | 0 (auto) | SM count step size. 0 = use architecture granularity. |
| `trials` | 5 | Number of independent trials; reports median. |

### Example

```bash
# Build for Blackwell
make CUDA_ARCH=sm_100

# Run with 1GB buffer, 30 iterations, 7 trials
./green_ctx_bw_bench 1024 30 0 0 7 > results.csv

# Generate plot
python3 plot_results.py results.csv
```

## Output

### CSV (stdout)

```
sm_count_requested,sm_count_allocated,bandwidth_GBps,pct_of_full_gpu
84,84,337.152,100.0
4,4,81.658,24.2
8,8,168.845,50.1
...
```

- **sm_count_requested**: The SM count passed to `cuDevSmResourceSplitByCount`.
- **sm_count_allocated**: The actual SM count allocated (may be rounded up by the driver).
- **bandwidth_GBps**: Measured DRAM read bandwidth in GB/s (median of trials).
- **pct_of_full_gpu**: Bandwidth as a percentage of the full-GPU baseline.

### Plot (`bw_saturation.png`)

Shows bandwidth vs. SM count with:
- Full-GPU baseline
- 95% saturation threshold
- Annotated saturation point

### Text Summary

```
============================================================
  Green Context BW Saturation Analysis
============================================================
  Full-GPU bandwidth:     337.15 GB/s
  Peak measured BW:       351.62 GB/s
  95% threshold:          320.29 GB/s
  Saturation point:       16 SMs (19.0% of 84 total)
  BW at saturation:       337.98 GB/s (100.2% of full-GPU)
============================================================
```

## File Structure

```
green_ctx_bw_bench/
├── green_ctx_bw_bench.cu   # Main CUDA benchmark source
├── Makefile                # Build system (default: sm_100)
├── run_bench.sh            # One-command build + run + plot
├── plot_results.py         # Analysis & visualization
└── README.md               # This file
```

## Notes

- The kernel uses `__ldg()` (read-only texture cache path) for streaming loads, which is the standard technique for bandwidth benchmarks.
- Green Contexts may sometimes allow work to run on more SMs than provisioned (documented CUDA behavior), but the median-of-trials approach mitigates measurement noise from this.
- Memory is allocated in the **primary context** and accessible from all Green Contexts on the same device (unified address space).
- The benchmark measures **read-only** bandwidth. Write bandwidth saturation may differ.
