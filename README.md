# Green Context DRAM Bandwidth Saturation Benchmark

Measure streaming-read bandwidth versus allocated SM count using CUDA Green Contexts on NVIDIA H200 and B200/GB200. The reported saturation point is the smallest sampled allocation reaching 95% of the full-GPU baseline for the selected kernel and configuration. Verify physical DRAM traffic with Nsight before interpreting it as a hardware limit.

## H200 or B200 on Runpod

Select the GPU and a **CUDA development image containing `nvcc`, a C++ compiler, and `make`**:

| GPU | Native build target | Minimum `nvcc` toolkit |
|---|---|---|
| H200 (including H200 NVL) | `sm_90` | CUDA 12.4 |
| B200 / GB200 | `sm_100` | CUDA 12.8 |

NVIDIA lists [H200 as compute capability 9.0 and B200 as 10.0](https://developer.nvidia.com/cuda/gpus). Green Contexts require [CUDA 12.4 or newer](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html). A CUDA 12.8+ development image works for either GPU. The CUDA version displayed by `nvidia-smi` describes driver compatibility; check `nvcc --version` for the actual compiler.

Use your home directory; `/workspace` and root access are not required. On a fresh pod:

```bash
cd "$HOME"
git clone https://github.com/shh8228/cuda-green-context-bandwidth-bench.git
cd cuda-green-context-bandwidth-bench
nvidia-smi
nvcc --version
command -v make g++ python3

# H200: check the environment, then run a shorter initial comparison
python3 run_h200_runpod.py --preflight-only
python3 run_h200_runpod.py --modes 1,4 --blocks-per-sm 8 --tma-stages 4 --iterations 10 --trials 3

# Full default sweep, including TMA depths 2, 4, and 8
python3 run_h200_runpod.py
```

If the repository is already cloned, enter that directory and use the runner from the updated source. For B200, use `python3 run_b200_runpod.py` with the same options. Both entry points detect the selected GPU and choose its build target and toolkit requirement.

If `nvcc` is missing, check whether `/usr/local/cuda/bin/nvcc` exists; if so, add `/usr/local/cuda/bin` to `PATH`. Otherwise, select a CUDA **development** image with the required compiler. A CUDA runtime image or a PyTorch installation alone may lack it. Do not rely on `apt install nvidia-cuda-toolkit` to provide the required version. The runner and text analysis use Python's standard library, so you can run them without `pip`, a virtual environment, or `python3-venv`. `matplotlib` and `numpy` are optional for PNG plots.

The default run uses a 2 GiB input and compares direct L1-bypass loads (mode 1) with TMA (mode 4), with 8 and 32 queued blocks per SM and TMA depths of 2, 4, and 8. It compiles a fresh native executable (`sm_90` on H200, `sm_100` on B200) inside a new results directory, pins the selected GPU by UUID, validates every load mode, then sweeps the supported SM allocations in randomized order. It repeats the full-GPU baseline afterward to reveal drift. Correctness checks also run inside each green context before timing.

For a shorter first run, or for the full validation tools:

```bash
# One TMA configuration
python3 run_h200_runpod.py --modes 4 --blocks-per-sm 8 --tma-stages 4

# Additional device instrumentation; requires matching Compute Sanitizer and ncu
python3 run_h200_runpod.py --sanitize --profile

# Select an NVML device index/UUID, buffer size, and a NEW output directory
python3 run_h200_runpod.py --gpu-id 0 --buffer-mb 4096 --output-dir "$HOME/h200-run-1"
```

`--sanitize` runs memcheck, racecheck, and synccheck on the built-in correctness cases before timing. `--profile` separately collects actual DRAM read/write bytes, kernel duration, and DRAM throughput at 8 SMs, the measured 95% point, and the full GPU for each configuration. Profiling requires permission to access performance counters on the Runpod host; `ERR_NVGPUCTRPERM` must be resolved by the host/provider. Profiler event timings are kept separate from ordinary benchmark timings.

Artifacts are saved under `h200_results/<UTC timestamp>/` or `b200_results/<UTC timestamp>/`, according to the detected GPU, or `--output-dir`:

- A fresh executable and copies/hashes of its source inputs.
- `manifest.json` with GPU UUID, driver/toolkit versions, arguments, commands, exit codes, and completion status.
- One bandwidth CSV and plot per configuration. Plotting needs `matplotlib` and `numpy`; CSV and text analysis work without them.
- Logs with correctness results, per-trial timings, compiled PTX target, register/shared-memory use, theoretical occupancy, residency probe counts, and baseline drift.
- Optional sanitizer logs and Nsight CSV files in base units.

An error stops the runner and leaves `complete: false` in the manifest. The built-in checks cover all output threads, partial tiles, idle CTAs, and repeated TMA barrier phases using varied integer input and a CPU reference. These checks do not replace running the sanitizer suite on the actual GPU.

A separate residency probe reports the SMs available to a context; it does not instrument the timed kernel or prove that every allocated SM was busy throughout it. Compare absolute bandwidth as well as each configuration's own 95% threshold. The L2-size check (input >= 4x L2) reduces cache contamination but does not prove that the counted bytes reached DRAM; use the profiler's `dram__bytes_read.sum` for that check.

### TMA consumer control: transfer/wait versus FP32 consumption

The ordinary TMA kernel transfers each 16 KiB tile into shared memory, then all
256 threads read its contents into registers and accumulate FP32 sums. To test
whether that consumer limits the rate at which buffers can be reused, add
`--tma-copy-only` to the runner or to mode 4 of the standalone benchmark.

Both paths use the **same compiled kernel** with a runtime-uniform consumer flag.
They retain the same tile mapping, pipeline depth, thread/block count, shared
allocation, register allocation, theoretical occupancy limit, TMA issue/wait
operations, two CTA barriers per tile, and per-thread sink writes. The control
skips tile reads and FP32 sums; each thread instead writes its block's completed
tile count. Every issued TMA transfer is still waited on before buffer reuse and
kernel completion. Reported bandwidth counts input bytes, not shared-memory or
sink traffic. This conservative control retains unused consumer resources and
does not measure an independently optimized TMA-only engine.

Validation first checks full payload sums using the consuming branch of that
same kernel, then separately checks the actual copy-only branch's completion
counts, including partial tiles, idle blocks and repeated barrier phases. The
payload checks execute outside timing; copy-only sink values do not themselves
check payload contents. Run Compute Sanitizer on the actual GPU before relying
on new device results.

For a paired full SM sweep on B200 (use new output directories):

```bash
python3 run_b200_runpod.py --modes 4 --blocks-per-sm 24 --tma-stages 2,4 \
  --iterations 20 --trials 7 --seed 8229 --output-dir b200_consume
python3 run_b200_runpod.py --modes 4 --blocks-per-sm 24 --tma-stages 2,4 \
  --iterations 20 --trials 7 --seed 8229 --tma-copy-only --output-dir b200_copy_only
```

Repeat in reversed order and compare absolute GB/s against one common full-GPU
reference, as well as each path's own full-GPU value. Monitor clocks, temperature,
power and baseline drift. A smaller threshold relative to a slower path's own
baseline does not establish better use of HBM.

If copy-only is faster at the same SM allocation, the removed shared-memory
reads, FP32 work, and their delay of future TMA issues limit the original loop.
This does **not** isolate shared-memory read ports, register delivery, arithmetic
dependencies, or scheduler issue overhead individually. If the curves coincide,
the consumer is not the dominant constraint under that configuration; this alone
does not identify a hardware request-queue limit. Neither result predicts Tensor
Core performance: a real WGMMA/TCGEN05 consumer has different operand layouts,
synchronization, accumulator storage, and useful computation.

### B200 Tensor Core consumer (mode 6)

Mode 6 adds an actual `tcgen05.mma.cta_group::1.kind::tf32` consumer. It requires
**CUDA 12.8+ and `sm_100a`**, including when `sm_100` already built successfully.
H200 is not supported for this mode. Existing modes retain their original targets.

Each TMA transfer streams a packed **A[128×32] TF32 tile (16 KiB)**. A small
**B[32×N]** matrix is initialized in shared memory once per CTA. Four K=8 MMAs
consume each A tile; outputs accumulate in TMEM across that CTA's input tiles,
then every output is read from TMEM and written to global memory once. No extra
MMA repetitions are added to inflate arithmetic work.

The default `--tc-schedule overlap` removes the CTA-wide barriers from MMA's
tile loop. Thread 0 waits for a tile's TMA, issues its four MMAs and commits to
that slot's completion barrier, then advances. Once the ring is occupied, it
retires the oldest slot **after issuing the current tile's MMA**, waits for that
old slot's reader to finish, and starts its replacement TMA. This permits MMA
and TMA work on different slots to overlap without overwriting live operands.
All outstanding MMAs are drained before the block joins for the epilogue.

`--tc-schedule serial` retains the per-tile MMA completion wait for comparison
inside the same compiled kernel. Both schedules now store MMA outputs in
column-major order (`output[n*128 + row]`), so adjacent warp lanes write adjacent
floats. Thus the new serial/overlap comparison isolates the scheduling change;
neither binary exactly reproduces the old strided output stores. Arithmetic,
streaming input bytes, TMEM allocation and output byte counts are unchanged.
The two-stage specialization uses a six-CTA launch-bound register budget, to
avoid losing residency just from the issuer's longer live ranges. This is a
compiler resource constraint, not a guarantee of six active CTAs at runtime.

| `--tc-n` | FLOPs per A tile | FLOPs per streamed input byte | Shared B per CTA |
|---|---:|---:|---:|
| 8 (default) | 65,536 | 4 | 1 KiB |
| 16 | 131,072 | 8 | 2 KiB |
| 32 | 262,144 | 16 | 4 KiB |

Three runtime branches of the **same compiled kernel** are selectable with
`--tc-consumer mma|sum|copy`. For a fixed N and stage count they reserve the same
registers, shared buffers, 32 TMEM columns (16 KiB), and global output footprint.
`sum` consumes all A elements through SIMT shared loads; `copy` waits for all
transfers without reading their payload. Both controls retain the TMEM allocation
and B setup. Their completion barrier uses a software arrival instead of MMA's
async commit, and their epilogue writes repeated sums/completion counts instead
of reading TMEM. These are matched-resource controls, not instruction-identical
workloads. They should be compared with mode-6 MMA, not silently substituted for
the older mode-4 controls.
The schedule option affects MMA only; copy/sum retain their per-tile control
loops. In particular they do not share overlap MMA's reduced CTA synchronization.
`tc_schedule` in control results records the requested MMA schedule (also used
for their out-of-timing payload validation), not a changed control loop.

**Run from the updated checkout on B200:**

```bash
# Build and check all modes; includes 54 TC shape/stage/consumer/schedule combinations.
make test-gpu CUDA_ARCH=sm_100a

# Small first check inside an 8-SM green context (default N=8, MMA).
./green_ctx_bw_bench --sm-count 8 --blocks-per-sm 24 --tma-stages 2 \
  --require-dram 2048 20 0 0 7 6

# Matched three-way SM sweep, lightest arithmetic first.
python3 run_b200_runpod.py --modes 6 --tc-n 8 --tc-consumers mma,sum,copy \
  --blocks-per-sm 24 --tma-stages 2,4 --iterations 20 --trials 7 \
  --output-dir b200_tc_n8

# Arithmetic-intensity sensitivity: 4, 8, 16 FLOPs/input byte.
python3 run_b200_runpod.py --modes 6 --tc-n 8,16,32 --tc-consumers mma,sum,copy \
  --blocks-per-sm 24 --tma-stages 2,4 --iterations 20 --trials 7 \
  --output-dir b200_tc_intensity

# Compare scheduling with the same compiler, epilogue and resource reservations.
# Run in reversed order in a second experiment to check thermal/clock drift.
for schedule in serial overlap; do
  python3 run_b200_runpod.py --modes 6 --tc-n 8 --tc-consumers mma,copy \
    --tc-schedule "$schedule" --blocks-per-sm 24 --tma-stages 2,4 \
    --iterations 20 --trials 7 --output-dir "b200_tc_${schedule}"
done
```

The runner selects `sm_100a` automatically when mode 6 is requested. Files include
the shape, consumer and schedule, e.g. `mode6_blocks24_stages2_n8_mma_overlap.csv`. The manifest
preserves the source, build target and arguments; CSV records also identify N,
consumer, `tc_schedule` and `mma_flops_per_input_byte` (zero for the non-MMA controls; this
does not count SIMT sums). Use a new output directory for each run.

Before relying on results, run the device validation under a matching Compute
Sanitizer (these tools do not use Nsight Compute performance counters):

```bash
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 1 \
    ./green_ctx_bw_bench --validate-only --tc-n 8 --tma-stages 2 1 1 0 0 1 6 || break
done
# Or add --sanitize to the runner to check every requested configuration.
```

Validation compares **every MMA output** with an independent CPU matrix reference
on varied exactly representable inputs, including partial tiles, idle CTAs and
multiple barrier phase wraps. The 33 matrix cases include exact ring boundaries
and partial tails around them. For sum/copy it additionally checks the selected
branch outside timing. Copy completion counts alone do not verify payload data.

The bandwidth numerator is only streamed A bytes. B initialization, TMEM
allocation, synchronization, computation and output stores are included in time;
their bytes are not added to that numerator. TF32 inputs are preformatted in the
TCGEN05 no-swizzle K-major layout (logical row `r`, reduction index `k` maps to
`(k/4)*512 + r*4 + k%4`), so there is no timed packing pass. Timing input values
have their low 13 mantissa bits zeroed for valid TF32 operands, consistently for
all three consumers. These format/data changes also limit direct comparison with
old mode-4 measurements.

Start with N=8, but **low FLOPs/byte does not prove an HBM bottleneck**. Small MMA
shapes, synchronization and the compute capacity of the allocated SMs can still
limit throughput. Compare absolute bandwidth at each SM allocation as N grows,
and compare the matched copy curve too. Increasing N also changes B shared size
and the output footprint; use each N's matched controls and resource logs to
interpret a drop. Use one common full-GPU reference when comparing saturation
thresholds, and repeat runs in reversed consumer order to assess drift (for
example `--tc-consumers copy,sum,mma`).

This is a consumer-path microbenchmark, **not a complete decode GEMM or serving
workload**: B is reused across tiles and accumulated output traffic is amortized.
It neither reproduces model/KV layouts and quantization nor concurrent prefill.
The implementation can be compiled and its host paths tested without a GPU;
correctness and performance of TCGEN05 require validation on the actual B200.

Implementation references: NVIDIA's [TCGEN05 PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)
and [CUTLASS TF32 shared/shared instruction wrappers](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_umma.hpp).

## Motivation

When spatially partitioning a GPU with CUDA Green Contexts (e.g., for disaggregated serving, MPS-like workload isolation, or co-located inference), a key question is: **how many SMs are actually needed to saturate memory bandwidth?** If a memory-bound kernel saturates DRAM bandwidth with only a fraction of available SMs, the remaining SMs can be allocated to compute-bound work without degrading memory throughput.

## How It Works

1. **Baseline**: Measures full-GPU effective read bandwidth for the selected kernel.
2. **Sweep**: For each SM count from the architecture minimum up to the total:
   - Creates a **Green Context** with that many SMs via `cuDevSmResourceSplitByCount` + `cuGreenCtxCreate`.
   - Launches a pure streaming-read kernel (`__ldg` float4 loads, coalesced access pattern) confined to the Green Context.
   - Measures achieved read bandwidth using CUDA events, taking the **median of multiple trials** for robustness.
3. **Analysis**: Identifies the smallest sampled SM allocation achieving ≥95% of that kernel/configuration’s full-GPU bandwidth.

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

- **Compiler**: CUDA ≥ 12.4 for H200 (`sm_90`); CUDA ≥ 12.8 for native B200/GB200 (`sm_100`). Other supported targets require CUDA ≥ 12.4. Use a development image with `nvcc`, a host C++ compiler, and `make`.
- **GPU**: H200 or B200 for the Runpod runner; the standalone benchmark also supports other Hopper and Ampere GPUs.
- **Python 3** with `matplotlib` (for plotting; analysis text output works without it)

## Build

```bash
# Default: Blackwell (sm_100)
make

# For other architectures:
make -B CUDA_ARCH=sm_90   # H200 / Hopper
make -B CUDA_ARCH=sm_86   # RTX A6000 (use sm_80 for A100)
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
| `buffer_MB` | 512 | Size of the read buffer in MiB. Use a size well above L2 capacity. |
| `iterations` | 20 | Kernel launches per timed trial. |
| `gpu_id` | 0 | CUDA device index. |
| `sm_step` | 0 (auto) | SM count step size. 0 = use the supported alignment (2 on Ampere, 8 on Hopper/B200). |
| `trials` | 5 | Number of independent trials; reports median. |
| `load_mode` | 0 | 0 = default `__ldg`, 1 = L1-bypass (`ld.global.cg`), 2 = `cp.async` staging, 3 = cooperative-groups async staging, 4 = TMA (`cp.async.bulk.tensor`, SM90+ only), 5 = decode-like KV-cache sweep, 6 = TCGEN05 TF32 (B200, `sm_100a`). |

### Load Modes

- `0`: Default streaming loads with `__ldg()`.
- `1`: L1-bypass path using `ld.global.cg`.
- `2`: Explicit `cp.async` shared-memory staging.
- `3`: Cooperative-groups async staging path, useful as the closest portable precursor to Hopper/Blackwell-style async copy flows.
- `4`: Hopper/Blackwell TMA path using a 2D tensor map and `cp.async.bulk.tensor.2d`.
- `5`: Lightweight transformer-decode-style KV-cache sweep that scans each token as K/V tiles and keeps the arithmetic intentionally small so memory bandwidth stays dominant.

Note: mode 4 requires both GPU and compiled PTX target >= SM90; mode 2 requires >= SM80. Unsupported builds fail instead of executing a no-op. TMA uses 16 KiB tiles with configurable pipeline depth, 128-byte-aligned shared memory, explicitly published barrier initialization, and protected buffer reuse.

Additional options work with the positional CLI:

```bash
./green_ctx_bw_bench --validate-only 1 1 0 0 1 4
./green_ctx_bw_bench --tma-stages 4 --blocks-per-sm 8 --require-dram 2048 30 0 0 7 4
./green_ctx_bw_bench --sm-count 8 --tma-stages 4 2048 30 0 0 7 4
make test CUDA_ARCH=sm_100
make test-gpu CUDA_ARCH=sm_100
```

`--sm-count` emits one JSON measurement. `--skip-validation` is intended for profiling after successful correctness checks. The CSV retains the original four columns and adds `probe_observed_sm_count`. All diagnostic output goes to stderr.

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
- **bandwidth_GBps**: Effective input read bandwidth in GB/s (median of trials); verify actual DRAM traffic with counters.
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
- Green Contexts can use extra SMs in specific documented MPS/CDP scenarios. The benchmark reports residency probes and rejects observed counts above the allocation; taking a median does not fix an incorrect SM-count assumption.
- Memory is allocated in the **primary context**, which is shared by the converted Green Contexts on the same device.
- The numerator counts input read bytes; kernels also write a small sink. Use profiler counters to distinguish effective read throughput from actual DRAM traffic.
