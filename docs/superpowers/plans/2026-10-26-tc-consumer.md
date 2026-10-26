# B200 Tensor Core consumer implementation plan

> Execute inline using superpowers:executing-plans, then request one fresh code review.

**Goal:** Compare TMA copy, SIMT sum and actual Tensor Core consumption at varying SM allocations and low arithmetic intensities.

**Architecture:** Add mode 6 with one runtime-selected consumer in the same compiled kernel. Stream packed TF32 A tiles (128×32, 16 KiB), keep B (32×N) resident per CTA, perform four K=8 TCGEN05 MMA operations per tile, accumulate into TMEM and validate every final output. N=8/16/32 gives 4/8/16 FLOPs per streamed input byte.

**Tech stack:** CUDA 12.8+, sm_100a, existing green-context runner, stdlib Python tests.

**Spec:** User requests in this conversation: actual Tensor Core path, memory-bound intent, B200 Runpod operation, rigorous correctness checks.

## Constraints and review focus

- Keep existing modes 0–5 operational on their supported architectures.
- Reject mode 6 when the loaded binary lacks sm_100a instructions; no fallback reporting fake performance.
- Match resource reservations, input bytes, tile ownership and output footprint among mode-6 consumers.
- Check partial tiles, idle CTAs, all stage parities and every matrix output; never equate a completed copy with verified payload.
- Audit packed matrix descriptors, TMEM warp addressing, proxy fences and buffer reuse ordering against NVIDIA PTX.
- Report effective input bandwidth; no unmeasured physical HBM bandwidth or actual decode claims.
- No local GPU is available: compile and host tests here; GPU validation and sanitizer commands must ship for Runpod.

## Tasks

- [x] Add failing CLI and GPU tests for mode 6, N restrictions, consumer restrictions and all stage/N/consumer combinations.
- [x] Implement the TCGEN05 kernel in `tma_tensor_core.cuh`, host integration and CPU reference validation in `green_ctx_bw_bench.cu`, CLI in `bench_options.h`.
- [x] Compile sm_100a, inspect emitted PTX, run host tests; compile legacy sm_100 and sm_90 paths.
- [x] Extend runner architecture selection, validation, filenames and metadata; add tests and a Runpod guide to README.
- [x] Request an independent code review, address material issues, run required verification and report the GPU execution limitation.

## Execution notes

- Work in the existing clean checkout; do not commit or push without user instruction.
- Proceed with the requested implementation directly; no approval checkpoint for reversible edits.
- CLI and runner tests were observed failing before implementation, then passing.
- Host suite: 59 tests, 54 passed, 5 device tests skipped. Native sm_100a and legacy sm_100/sm_90 builds succeeded.
- PTX contains the expected TCGEN05 MMA, allocation, load and completion instructions. All nine TC variants compile without register spills.
- Independent review found no actionable correctness defects. Actual B200 validation and Compute Sanitizer remain unexecuted; README supplies both commands and limitations.
