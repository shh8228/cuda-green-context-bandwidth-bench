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
./green_ctx_bw_bench [buffer_MB] [iterations] [gpu_id] [sm_step] [trials] [load_mode]
```

| Argument | Default | Description |
|---|---|---|
| `buffer_MB` | 512 | Size of the read buffer in MB. Larger = more stable results. |
| `iterations` | 20 | Kernel launches per timed trial. |
| `gpu_id` | 0 | CUDA device index. |
| `sm_step` | 0 (auto) | SM count step size. 0 = use architecture granularity. |
| `trials` | 5 | Number of independent trials; reports median. |
| `load_mode` | 0 | 0 = default `__ldg`, 1 = L1-bypass (`ld.global.cg`), 2 = `cp.async` staging, 3 = cooperative-groups async staging, 4 = TMA (`cp.async.bulk.tensor`, SM90+ only), 5 = decode-like KV-cache sweep. |

### Load Modes

- `0`: Default streaming loads with `__ldg()`.
- `1`: L1-bypass path using `ld.global.cg`.
- `2`: Explicit `cp.async` shared-memory staging.
- `3`: Cooperative-groups async staging path, useful as the closest portable precursor to Hopper/Blackwell-style async copy flows.
- `4`: Hopper/Blackwell TMA path using a 2D tensor map and `cp.async.bulk.tensor.2d`.
- `5`: Lightweight transformer-decode-style KV-cache sweep that scans each token as K/V tiles and keeps the arithmetic intentionally small so memory bandwidth stays dominant.

Note: `load_mode=4` requires compute capability 9.0 or higher.

### Example

```bash
# Build for Blackwell
make CUDA_ARCH=sm_100

# Run with 1GB buffer, 30 iterations, 7 trials
./green_ctx_bw_bench 1024 30 0 0 7 > results.csv

# Generate plot
python3 plot_results.py results.csv
```

## Per-GPC Global-Memory Bottleneck Test

`run_gpc_sweep.py` compares two equal-size TPC masks while running only the
direct global-load path (`ld.global.cg` by default):

- **packed**: fills the largest GPCs first, minimizing the number of GPCs used;
- **scattered**: selects TPCs round-robin across GPCs, maximizing GPC spread.

If scattered placement is repeatedly faster at the same TPC count and the
same observed SM count, that is evidence for a per-GPC global-memory injection,
L2-partition, or NoC bottleneck. This test does not use shared memory or DSM.

### Additional prerequisites

Build the `ecrts25-ae` libsmctrl artifact so its directory contains
`nvtaskset`, `libcuda.so.1`, and `libsmctrl_test_gpc_info`. The topology query
also requires NVIDIA's `nvdebug` kernel module.

```bash
cd /path/to/libsmctrl
make all

cd /path/to/green_ctx_bw_bench
make clean
make CUDA_ARCH=sm_100
make test CUDA_ARCH=sm_100
```

Capture the physical topology while `nvdebug` is loaded, then unload it before
starting MPS. This avoids a driver write-lock conflict observed when
`nvdebug` and `nvidia-cuda-mps-server` overlap:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 run_gpc_sweep.py \
  --libsmctrl-dir /path/to/libsmctrl \
  --gpu-id 0 \
  --capture-topology-json b200_topology_input.json

sudo rmmod nvdebug
```

Run the automatic sweep from the saved map. On a normal one-GPU Runpod, the
physical and process-visible device are both ordinal 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu0 \
python3 run_gpc_sweep.py \
  --libsmctrl-dir /path/to/libsmctrl \
  --topology-json b200_topology_input.json \
  --benchmark ./green_ctx_bw_bench \
  --gpu-id 0 \
  --buffer-mb 1024 \
  --iterations 30 \
  --trials 5 \
  --repetitions 7 \
  --load-mode 1 \
  --tpc-counts auto \
  --output-prefix b200_gpc_bw
```

To target physical GPU 1 on a multi-GPU host, capture with `--gpu-id 1`, then
run with `CUDA_VISIBLE_DEVICES=1` and `--gpu-id 0`; CUDA renumbers the sole
visible GPU to process ordinal 0. Give each isolated GPU a distinct
`CUDA_MPS_PIPE_DIRECTORY`. The runner sends raw hexadecimal masks to
`nvtaskset`, avoiding its hard-coded GPU-0 validation path while preserving
the saved physical GPC mapping.

`auto` tests whole-GPC-sized steps through half of the GPU and explicitly adds
the 40% and 50% TPC points. A custom list such as `--tpc-counts 16,24,32,40`
can be supplied instead. Each packed/scattered pair is run in randomized order.

The run produces:

- `b200_gpc_bw_raw.csv`: every timed process, exact masks, GPC spans, run order,
  bandwidth, and observed SM IDs;
- `b200_gpc_bw_summary.csv`: median packed/scattered bandwidth, their ratio,
  percentage gain, and number of paired scattered wins;
- `b200_gpc_bw_topology.json`: parsed topology, the original libsmctrl output,
  seed, and load mode.

The runner aborts if an equal-TPC packed/scattered pair observes different SM
counts, or if a mask exposes fewer than one or more than two SMs per TPC. As a
practical signal, look for a scattered/packed ratio above 1.05
that is consistent across most or all seven pairs. A ratio near 1.00 argues
against a per-GPC bottleneck and points instead toward per-SM request issue or
outstanding-request limits.

Run on an otherwise idle GPU with stable application clocks and power state.
`bandwidth_GBps` is useful bytes divided by CUDA-event time. Use a buffer well
above L2 capacity (the example uses 1 GiB), and confirm any winning placement
with Nsight Compute's `dram__bytes_read.sum.per_second` counter before treating
the difference as an HBM-bandwidth result.

### Important B200 validation caveat

The artifact's QMD-v4 support is not established on B200, and Hopper-or-newer
mask-bit numbering may not match software-visible TPC numbering. The `%smid`
probe verifies equal visible SM exposure; it cannot by itself prove that the
requested masks correspond to the intended physical GPCs under MPS. Preserve
the topology JSON and treat a placement result as provisional until the B200
mask-to-GPC mapping is independently validated.

## Raw-Mask Placement Test Without `nvdebug`

When `/proc/gpu0/num_gpcs` is unavailable, `run_raw_mask_sweep.py` tests the
weaker but still useful hypothesis that physical mask selection affects global
memory bandwidth at a fixed active-SM count. It does not identify GPCs.

The runner first enables all 128 raw mask bits. It then removes one bit at a
time and records the decrease in observed SM count. Ineffective bits are
discarded, and the largest population with a uniform one- or two-SM decrease
per bit is used for randomized fixed-cardinality masks. Every timed mask must
expose exactly the expected number of SMs or the run aborts.

Use `--mask-layouts packed,scattered` for a controlled low-TPC comparison.
Packed masks use adjacent positions in the discovered effective-bit list;
scattered masks maximize separation in that list. This is still a raw-index
placement comparison, not proof that the scattered masks cross GPCs.

This mode needs `nvtaskset` and its `libcuda.so.1` wrapper but does not need
`libsmctrl_test_gpc_info`, `nvdebug`, or `/proc/gpu0/num_gpcs`.

```bash
cd /path/to/green_ctx_bw_bench

python3 run_raw_mask_sweep.py \
  --libsmctrl-dir /path/to/libsmctrl \
  --benchmark ./green_ctx_bw_bench \
  --candidate-bits 128 \
  --probe-timeout 30 \
  --buffer-mb 1024 \
  --iterations 30 \
  --trials 5 \
  --repetitions 3 \
  --masks-per-count 12 \
  --load-mode 1 \
  --tpc-counts auto \
  --seed 8228 \
  --output-prefix b200_raw_mask_bw
```

For the paired low-TPC experiment:

```bash
python3 run_raw_mask_sweep.py \
  --libsmctrl-dir /path/to/libsmctrl \
  --benchmark ./green_ctx_bw_bench \
  --candidate-bits 128 \
  --mask-layouts packed,scattered \
  --tpc-counts 2,4,6,8 \
  --masks-per-count 4 \
  --repetitions 3 \
  --buffer-mb 1024 \
  --iterations 30 \
  --trials 5 \
  --load-mode 1 \
  --seed 8228 \
  --output-prefix b200_low_tpc_layout
```

The leave-one-out discovery phase starts 129 short CUDA processes. The timed
phase runs 12 masks per active-TPC count, with three process-level repetitions
per mask, in randomized order. Both phases checkpoint their CSV files after
each completed process.

Outputs:

- `b200_raw_mask_bw_probe.csv`: raw-bit effectiveness map;
- `b200_raw_mask_bw_raw.csv`: every timed measurement;
- `b200_raw_mask_bw_masks.csv`: median and repeat noise for each mask;
- `b200_raw_mask_bw_summary.csv`: bandwidth spread across equal-size masks;
- `b200_raw_mask_bw_comparison.csv`: packed-versus-scattered median bandwidth
  and relative gain (written when both layouts are requested);
- `b200_raw_mask_bw_metadata.json`: selected bits, SMs per bit, seed, and the
  explicit `raw-mask placement sensitivity; no GPC identity` claim scope.

Compare `between_mask_spread_pct` with `median_within_mask_cv_pct`. A between-
mask spread above 5% that is several times larger than repeat noise indicates
placement sensitivity. It does not prove the fast masks span more GPCs; that
requires an independently validated physical topology map.

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
