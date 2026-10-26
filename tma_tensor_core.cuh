#pragma once

// Standalone PTX implementation, no CUTLASS dependency.
// NVIDIA PTX ISA, tcgen05: K-major canonical no-swizzle layout,
// shared/instruction descriptors, TMEM allocation and 32x32b load mapping.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions
// A: M=128, K=32 TF32; B: K=32, N=8/16/32 TF32; D: M x N FP32.
// Four K=8 MMAs consume each 16 KiB A tile once. B is initialized once/CTA.

enum class TcConsumer : int { Mma = 0, Sum = 1, Copy = 2 };

__host__ __device__ constexpr float tc_b_value(int k, int n)
{
    return static_cast<float>((17 * k + 13 * n + 3 * k * n) % 127 - 63) / 64.0f;
}

// Check the architecture of the loaded image, including manually built/fat binaries.
__device__ __constant__ int tcgen05_image_supported =
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
    1;
#else
    0;
#endif

#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
__device__ __forceinline__ uint64_t tc_smem_desc(const void *ptr, unsigned leading_bytes)
{
    const uint64_t base = __cvta_generic_to_shared(ptr);
    // Matrix starts and both strides are 16-byte aligned. No swizzle,
    // relative leading stride, fixed descriptor version field = 001.
    return ((base & 0x3ffffULL) >> 4) |
           (static_cast<uint64_t>(leading_bytes >> 4) << 16) |
           (uint64_t(128 >> 4) << 32) | (1ULL << 46);
}

__device__ __forceinline__ void tc_mma(uint32_t dst, uint64_t a, uint64_t b,
                                      uint32_t descriptor, bool accumulate)
{
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, %5, %5, %5}, p; }"
        :: "r"(dst), "l"(a), "l"(b), "r"(descriptor), "r"(int(accumulate)), "r"(0) : "memory");
}

__device__ __forceinline__ void tc_load8(uint32_t address, float (&values)[8])
{
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(address) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = __uint_as_float(r[i]);
}
#endif

template<int Stages, int N>
// Two-stage shared buffers fit six CTAs on B200. Keep the async issuer's
// register live ranges from unnecessarily reducing that residency budget.
__global__ void __launch_bounds__(256, Stages == 2 ? 6 : 1)
read_bandwidth_kernel_tc(float *sink, size_t n_float4,
                         const CUtensorMap *tensor_map, TcConsumer consumer,
                         bool overlap)
{
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
    static_assert(N == 8 || N == 16 || N == 32, "supported N");
    extern __shared__ __align__(128) unsigned char tc_storage[];
    // Reserve identical resources for all three runtime-selected consumers.
    float *b = reinterpret_cast<float *>(tc_storage + Stages * kTmaTileBytes);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bars[Stages];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> mma_done[Stages];
    __shared__ uint32_t tmem;
    for (int i = threadIdx.x; i < N * 32; i += blockDim.x) {
        // Decode canonical packed B offset independently of A's descriptor.
        int k = (i / (N * 4)) * 4 + i % 4;
        int n = (i % (N * 4)) / 4;
        b[i] = tc_b_value(k, n);
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < Stages; ++s) {
            init(&bars[s], 1);
            init(&mma_done[s], 1);
        }
    }
    // Each writer publishes its generic shared writes to async consumers.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (threadIdx.x < 32) {
        uint32_t addr = __cvta_generic_to_shared(&tmem);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                     :: "r"(addr) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const size_t tiles = (n_float4 + kTmaTileFloat4s - 1) / kTmaTileFloat4s;
    auto issue_tile = [&](int stage, size_t tile) {
        (void)cuda::device::barrier_arrive_tx(bars[stage], 1, kTmaTileBytes);
        uint32_t dst = __cvta_generic_to_shared(tc_storage + stage * kTmaTileBytes);
        uint32_t bar = __cvta_generic_to_shared(cuda::device::barrier_native_handle(bars[stage]));
        int row = static_cast<int>(tile * kTmaRowsPerTile);
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {0, %2}], [%3];"
                     :: "r"(dst), "l"(tensor_map), "r"(row), "r"(bar) : "memory");
    };
    if (threadIdx.x == 0)
        for (int s = 0; s < Stages; ++s)
            if (blockIdx.x + size_t(s) * gridDim.x < tiles)
                issue_tile(s, blockIdx.x + size_t(s) * gridDim.x);

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t iteration = 0;
    constexpr uint32_t idesc = (1U << 4) | (2U << 7) | (2U << 10) |
                               ((N / 8) << 17) | (8U << 24);
    if (consumer == TcConsumer::Mma && overlap) {
        // All threads need this uniform count for the warp-collective epilogue.
        iteration = blockIdx.x < tiles ? 1 + (tiles - 1 - blockIdx.x) / gridDim.x : 0;
        if (threadIdx.x == 0) {
            // One issuer controls both async engines; other threads join only
            // after the ring has drained. Same-shape MMA accumulation is pipelined.
            uint64_t b_desc[4];
            #pragma unroll
            for (int slice = 0; slice < 4; ++slice)
                b_desc[slice] = tc_smem_desc(b + slice * N * 8, N * 16);
            for (size_t i = 0; i < iteration; ++i) {
                int stage = i % Stages;
                bars[stage].wait_parity((i / Stages) & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                const float *a = reinterpret_cast<const float *>(tc_storage + stage * kTmaTileBytes);
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    tc_mma(tmem, tc_smem_desc(a + slice * 1024, 2048), b_desc[slice],
                           idesc, i != 0 || slice != 0);
                uint32_t bar = __cvta_generic_to_shared(cuda::device::barrier_native_handle(mma_done[stage]));
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                             :: "r"(bar) : "memory");

                // Launch this tile before waiting on the oldest slot. Thus its
                // MMA can overlap the oldest-slot wait and the following TMA.
                if (i >= Stages - 1) {
                    size_t retired = i - (Stages - 1);
                    int slot = retired % Stages;
                    mma_done[slot].wait_parity((retired / Stages) & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    if (retired + Stages < iteration)
                        issue_tile(slot, blockIdx.x + (retired + Stages) * gridDim.x);
                }
            }
            // Short CTAs may never have retired a slot. Wait every remaining
            // commit, not a guessed last-slot parity, before any TMEM loads.
            size_t drained = iteration >= Stages ? iteration - Stages + 1 : 0;
            for (size_t i = drained; i < iteration; ++i)
                mma_done[i % Stages].wait_parity((i / Stages) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        }
        __syncthreads();
    } else {
        // Serial MMA baseline and the original SIMT/copy control schedules.
        for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x, ++iteration) {
            int stage = iteration % Stages;
            if (threadIdx.x == 0) bars[stage].wait_parity((iteration / Stages) & 1);
            __syncthreads();
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            const float *a = reinterpret_cast<const float *>(tc_storage + stage * kTmaTileBytes);
            if (consumer == TcConsumer::Mma && threadIdx.x == 0) {
                #pragma unroll
                for (int slice = 0; slice < 4; ++slice)
                    tc_mma(tmem, tc_smem_desc(a + slice * 1024, 2048),
                           tc_smem_desc(b + slice * N * 8, N * 16), idesc,
                           iteration != 0 || slice != 0);
            } else if (consumer == TcConsumer::Sum) {
                const float4 *v = reinterpret_cast<const float4 *>(a);
                for (size_t i = threadIdx.x; i < kTmaTileFloat4s; i += blockDim.x) {
                    // TMA zero-fills the tail, but match the original SUM's guard.
                    if (tile * kTmaTileFloat4s + i < n_float4) {
                        float4 x = v[i];
                        accum.x += x.x; accum.y += x.y; accum.z += x.z; accum.w += x.w;
                    }
                }
            }
            if (threadIdx.x == 0) {
                if (consumer == TcConsumer::Mma) {
                    uint32_t bar = __cvta_generic_to_shared(cuda::device::barrier_native_handle(mma_done[0]));
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                                 :: "r"(bar) : "memory");
                } else {
                    (void)mma_done[0].arrive();
                }
                mma_done[0].wait_parity(iteration & 1);
                asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            }
            // No TMA stage overwrite until MMA has finished reading both operands
            // and all SIMT readers (if selected) have finished.
            __syncthreads();
            if (threadIdx.x == 0 && tile + size_t(Stages) * gridDim.x < tiles)
                issue_tile(stage, tile + size_t(Stages) * gridDim.x);
        }
    }
    float *out = sink + size_t(blockIdx.x) * 128 * N;
    if (consumer == TcConsumer::Mma && iteration != 0) {
        // Warp w in the first warpgroup accesses TMEM rows [32w,32w+31].
        if (threadIdx.x < 128) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            const uint32_t row_base = (threadIdx.x / 32) * 32;
            #pragma unroll
            for (int col = 0; col < N; col += 8) {
                float v[8];
                tc_load8(tmem + (row_base << 16) + col, v);
                #pragma unroll
                // Column-major output: adjacent lanes write adjacent floats.
                // Applied to both schedules, so their comparison isolates the loop.
                for (int j = 0; j < 8; ++j) out[(col + j) * 128 + threadIdx.x] = v[j];
            }
        }
    } else {
        float value = consumer == TcConsumer::Sum ? accum.x + accum.y + accum.z + accum.w
                                                 : static_cast<float>(iteration);
        for (int i = threadIdx.x; i < 128 * N; i += blockDim.x) out[i] = value;
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    // Every TMEM reader has completed its tcgen05.wait::ld before deallocation.
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (threadIdx.x < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                     :: "r"(tmem) : "memory");
#else
    __trap();
#endif
}
