/**
 * green_ctx_bw_bench.cu
 *
 * Microbenchmark: Minimum SMs to Saturate GPU DRAM Read Bandwidth
 * using CUDA Green Contexts on NVIDIA Blackwell (GB200).
 *
 * Methodology:
 *   For each SM count in {granularity, 2*granularity, ..., total_SMs}:
 *     1. Create a Green Context with that many SMs.
 *     2. Launch a memory-read-only kernel (streaming loads) inside
 *        the green context, using enough threads/blocks to keep
 *        all assigned SMs busy.
 *     3. Measure achieved read bandwidth via CUDA events.
 *        Uses median of multiple trials for robustness.
 *   The "saturation point" is the smallest SM count where bandwidth
 *   reaches >= 95% of the full-GPU baseline bandwidth.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_100 green_ctx_bw_bench.cu -o green_ctx_bw_bench -lcuda
 *
 * Run:
 *   ./green_ctx_bw_bench [buffer_size_MB] [iterations] [gpu_id] [sm_step] [trials] [use_l1_bypass]
 *
 * Output: CSV to stdout with columns:
 *   sm_count_requested, sm_count_allocated, bandwidth_GBps, pct_of_full_gpu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
 
enum class LoadMode {
    Default = 0,
    L1Bypass = 1,
    CpAsync = 2,
    PipelineAsync = 3,
    Tma = 4,
    DecodeLike = 5,
};

static constexpr size_t kTmaRowBytes = 256;
static constexpr size_t kTmaTransferBytes = 16384;
static constexpr size_t kTmaStageBytes = 32768;
static constexpr size_t kTmaStages = 2;
static constexpr size_t kTmaTransfersPerStage = kTmaStageBytes / kTmaTransferBytes;
static constexpr size_t kTmaRowsPerTransfer = kTmaTransferBytes / kTmaRowBytes;
static constexpr size_t kTmaRowsPerStage = kTmaStageBytes / kTmaRowBytes;
static constexpr size_t kTmaFloat4sPerRow = kTmaRowBytes / sizeof(float4);
static constexpr size_t kTmaStageFloat4s = kTmaStageBytes / sizeof(float4);
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;
static constexpr size_t kTmaBarrierBytes = kTmaStages * sizeof(TmaBarrier);
static constexpr size_t kTmaDynamicSmemBytes =
    kTmaStages * kTmaStageBytes + kTmaBarrierBytes;
static_assert(kTmaTransferBytes <= 16384, "A single TMA transfer cannot exceed 16 KiB");
static_assert(kTmaStageBytes % kTmaTransferBytes == 0,
              "A TMA stage must contain whole transfers");
static_assert(kTmaTransferBytes % kTmaRowBytes == 0,
              "A TMA transfer must contain whole rows");
static constexpr size_t kDecodeTileBytes = 4096;
static constexpr size_t kDecodeTileFloat4s = kDecodeTileBytes / sizeof(float4);
static constexpr size_t kDecodeKvFloat4s = kDecodeTileFloat4s / 2;

static LoadMode g_load_mode = LoadMode::Default;

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_DRIVER_CHECK(call)                                              \
    do {                                                                     \
        CUresult err = (call);                                               \
        if (err != CUDA_SUCCESS) {                                           \
            const char *errStr = nullptr;                                    \
            cuGetErrorString(err, &errStr);                                  \
            fprintf(stderr, "CUDA Driver Error at %s:%d — %s: %s\n",        \
                    __FILE__, __LINE__, #call, errStr ? errStr : "unknown"); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDA_RT_CHECK(call)                                                  \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Runtime Error at %s:%d — %s: %s\n",       \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Streaming read kernel — pure memory-bandwidth bound
// Each thread reads 16 bytes (float4) per iteration in a coalesced pattern.
// We accumulate into a thread-local variable and write ONE value at the end
// to prevent the compiler from optimizing away the reads.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
read_bandwidth_kernel(const float4 *__restrict__ src,
                      float *__restrict__ sink,
                      size_t n_float4)
{
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (size_t i = tid; i < n_float4; i += stride) {
        float4 v = __ldg(&src[i]);   // read-only cached load
        accum.x += v.x;
        accum.y += v.y;
        accum.z += v.z;
        accum.w += v.w;
    }

    // Single coalesced write to prevent dead-code elimination
    if (tid < stride)
        sink[tid] = accum.x + accum.y + accum.z + accum.w;
}

// ---------------------------------------------------------------------------
// L1-bypass variant (non-cacheable loads) using inline PTX
// Uses ld.global.nc.v4.f32 to request non-cacheable global loads.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float4 load_nc_f4(const float4 *addr)
{
    float4 out;
    asm volatile(
        "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
        : "l"(addr));
    return out;
}

__device__ __forceinline__ void cp_async_16B(void *smem_ptr, const void *gmem_ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(smem), "l"(gmem_ptr), "n"(16));
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait0()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_cp_async(const float4 *__restrict__ src,
                               float *__restrict__ sink,
                               size_t n_float4)
{
    extern __shared__ float4 smem[];

    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (size_t i = tid; i < n_float4; i += stride) {
        cp_async_16B(&smem[threadIdx.x], &src[i]);
        cp_async_commit();
        cp_async_wait0();
        __syncthreads();

        float4 v = smem[threadIdx.x];
        accum.x += v.x;
        accum.y += v.y;
        accum.z += v.z;
        accum.w += v.w;
    }

    if (tid < stride)
        sink[tid] = accum.x + accum.y + accum.z + accum.w;
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_pipeline(const float4 *__restrict__ src,
                               float *__restrict__ sink,
                               size_t n_float4)
{
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    extern __shared__ float4 smem[];

    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (size_t i = tid; i < n_float4; i += stride) {
        cg::memcpy_async(block, &smem[threadIdx.x], &src[i], sizeof(float4));
        cg::wait(block);

        float4 v = smem[threadIdx.x];
        accum.x += v.x;
        accum.y += v.y;
        accum.z += v.z;
        accum.w += v.w;
    }

    if (tid < stride)
        sink[tid] = accum.x + accum.y + accum.z + accum.w;
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_tma(const float4 *__restrict__ src,
                          float *__restrict__ sink,
                          size_t n_float4,
                          const CUtensorMap *tensor_map)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) unsigned char smem_raw[];
    unsigned char *smem_tiles_raw = smem_raw;
    TmaBarrier *bars = reinterpret_cast<TmaBarrier *>(
        smem_raw + kTmaStages * kTmaStageBytes);
    unsigned char *smem_tiles[2] = {
        smem_tiles_raw,
        smem_tiles_raw + kTmaStageBytes,
    };
    float4 *tiles[2] = {
        reinterpret_cast<float4 *>(smem_tiles_raw),
        reinterpret_cast<float4 *>(smem_tiles_raw + kTmaStageBytes),
    };

    if (threadIdx.x == 0) {
        init(&bars[0], 1);
        init(&bars[1], 1);
    }
    __syncthreads();

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
    const size_t tile_bytes = kTmaStageBytes;
    const size_t tile_rows = kTmaRowsPerStage;
    const size_t float4s_per_row = kTmaFloat4sPerRow;
    const size_t tile_stride_rows = (size_t)gridDim.x * tile_rows;
    cuda::barrier<cuda::thread_scope_block>::arrival_token pending_token;
    bool has_pending_tile = false;

    auto issue_tile = [&](int stage, size_t tile_row) {
        auto token = cuda::device::barrier_arrive_tx(bars[stage], 1, tile_bytes);
        #pragma unroll
        for (int transfer = 0; transfer < kTmaTransfersPerStage; ++transfer) {
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                smem_tiles[stage] + transfer * kTmaTransferBytes,
                tensor_map,
                0,
                static_cast<int>(tile_row + transfer * kTmaRowsPerTransfer),
                bars[stage]);
        }
        return token;
    };

    size_t tile_row = blockIdx.x * tile_rows;
    size_t next_tile_row = tile_row + tile_stride_rows;
    int stage = 0;
    int next_stage = 1;

    if (tile_row * float4s_per_row < n_float4 && threadIdx.x == 0) {
        pending_token = issue_tile(stage, tile_row);
        bars[stage].wait(std::move(pending_token));
    }

    while (tile_row * float4s_per_row < n_float4) {
        __syncthreads();

        if (threadIdx.x == 0 && next_tile_row * float4s_per_row < n_float4) {
            pending_token = issue_tile(next_stage, next_tile_row);
            has_pending_tile = true;
        }

        __syncthreads();

        for (size_t i = threadIdx.x; i < kTmaStageFloat4s; i += blockDim.x) {
            size_t global_float4 = tile_row * float4s_per_row + i;
            if (global_float4 < n_float4) {
                float4 v = tiles[stage][i];
                accum.x += v.x;
                accum.y += v.y;
                accum.z += v.z;
                accum.w += v.w;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0 && has_pending_tile) {
            bars[next_stage].wait(std::move(pending_token));
            has_pending_tile = false;
        }

        __syncthreads();

        tile_row = next_tile_row;
        next_tile_row += tile_stride_rows;
        stage ^= 1;
        next_stage ^= 1;
    }

    if (threadIdx.x == 0) {
        sink[blockIdx.x] = accum.x + accum.y + accum.z + accum.w;
    }
#else
    (void)src;
    (void)sink;
    (void)n_float4;
    (void)tensor_map;
#endif
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_decode_like(const float4 *__restrict__ src,
                                  float *__restrict__ sink,
                                  size_t n_float4)
{
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t token_count = n_float4 / kDecodeTileFloat4s;

    float accum = 0.0f;
    const float q_scale = 1.0f + 0.01f * static_cast<float>(blockIdx.x & 31);

    for (size_t token = blockIdx.x; token < token_count; token += gridDim.x) {
        const size_t token_base = token * kDecodeTileFloat4s;
        float token_accum = 0.0f;

        // Lightweight attention-like KV-cache sweep: read K and V halves,
        // apply a tiny per-block scale, and keep the math cheap so bandwidth
        // remains the bottleneck.
        for (size_t i = threadIdx.x; i < kDecodeKvFloat4s; i += blockDim.x) {
            float4 k = __ldg(&src[token_base + i]);
            float4 v = __ldg(&src[token_base + kDecodeKvFloat4s + i]);
            float k_sum = k.x + k.y + k.z + k.w;
            float v_sum = v.x + v.y + v.z + v.w;
            token_accum += k_sum * q_scale + v_sum * 0.125f;
        }

        accum += token_accum;
    }

    if (tid < stride) {
        sink[tid] = accum;
    }
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_nc(const float4 *__restrict__ src,
                         float *__restrict__ sink,
                         size_t n_float4)
{
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

    for (size_t i = tid; i < n_float4; i += stride) {
        float4 v = load_nc_f4(&src[i]);
        accum.x += v.x;
        accum.y += v.y;
        accum.z += v.z;
        accum.w += v.w;
    }

    if (tid < stride)
        sink[tid] = accum.x + accum.y + accum.z + accum.w;
}

// ---------------------------------------------------------------------------
// Helper: compute median of a vector
// ---------------------------------------------------------------------------
static double median(std::vector<double> &v)
{
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::sort(v.begin(), v.end());
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

struct TmaContext {
    CUtensorMap *d_tensor_map = nullptr;
    bool enabled = false;
};

static const char *load_mode_name(LoadMode mode)
{
    switch (mode) {
    case LoadMode::Default: return "Default (__ldg)";
    case LoadMode::L1Bypass: return "L1-bypass (ld.global.cg)";
    case LoadMode::CpAsync: return "cp.async staging";
    case LoadMode::PipelineAsync: return "pipeline async staging";
    case LoadMode::Tma: return "TMA (cp.async.bulk.tensor)";
    case LoadMode::DecodeLike: return "Decode-like KV-cache sweep";
    }
    return "Unknown";
}

static bool uses_shared_staging(LoadMode mode)
{
    return mode == LoadMode::CpAsync || mode == LoadMode::PipelineAsync;
}

static bool uses_tma(LoadMode mode)
{
    return mode == LoadMode::Tma;
}

static bool build_tma_context(CUdevice cuDev, const float4 *d_src, size_t buf_bytes,
                              TmaContext *tma_ctx)
{
    int cc_major = 0;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&cc_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDev));
    if (cc_major < 9 || tma_ctx == nullptr) {
        return false;
    }

    CUtensorMap host_map{};
    const cuuint64_t global_dim[2]     = { kTmaRowBytes, buf_bytes / kTmaRowBytes };
    const cuuint64_t global_strides[1] = { kTmaRowBytes };            // dim1 간 stride = 256 bytes
    const cuuint32_t box_dim[2]        = { kTmaRowBytes, kTmaRowsPerTransfer };  // {256, 64}
    const cuuint32_t element_strides[2] = { 1, 1 };

    CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(&host_map,
                                             CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                             2,
                                             const_cast<float4 *>(d_src),
                                             global_dim,
                                             global_strides,
                                             box_dim,
                                             element_strides,
                                             CU_TENSOR_MAP_INTERLEAVE_NONE,
                                             CU_TENSOR_MAP_SWIZZLE_NONE,
                                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap *d_map = nullptr;
    CUDA_RT_CHECK(cudaMalloc(&d_map, sizeof(CUtensorMap)));
    CUDA_RT_CHECK(cudaMemcpy(d_map, &host_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    tma_ctx->d_tensor_map = d_map;
    tma_ctx->enabled = true;
    return true;
}

static void launch_selected_kernel(const float4 *d_src, float *d_sink,
                                   size_t n_float4, int blocks, int threads,
                                   cudaStream_t stream,
                                   const TmaContext &tma_ctx)
{
    const size_t shared_bytes = uses_shared_staging(g_load_mode)
        ? (size_t)threads * sizeof(float4)
        : 0;

    switch (g_load_mode) {
    case LoadMode::Default:
        read_bandwidth_kernel<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
        break;
    case LoadMode::L1Bypass:
        read_bandwidth_kernel_nc<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
        break;
    case LoadMode::CpAsync:
        read_bandwidth_kernel_cp_async<<<blocks, threads, shared_bytes, stream>>>(d_src, d_sink, n_float4);
        break;
    case LoadMode::PipelineAsync:
        read_bandwidth_kernel_pipeline<<<blocks, threads, shared_bytes, stream>>>(d_src, d_sink, n_float4);
        break;
    case LoadMode::Tma:
        read_bandwidth_kernel_tma<<<blocks, 256, kTmaDynamicSmemBytes, stream>>>(
            d_src, d_sink, n_float4, tma_ctx.d_tensor_map);
        break;
    case LoadMode::DecodeLike:
        read_bandwidth_kernel_decode_like<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
        break;
    }
}

// ---------------------------------------------------------------------------
// Measure bandwidth with a given stream, returning median over trials
// ---------------------------------------------------------------------------
static double measure_bandwidth_timed(cudaStream_t stream,
                                      const float4 *d_src, float *d_sink,
                                      size_t n_float4, int blocks, int iters,
                                      int trials, size_t buf_bytes,
                                      const TmaContext &tma_ctx)
{
    cudaEvent_t start, stop;
    CUDA_RT_CHECK(cudaEventCreate(&start));
    CUDA_RT_CHECK(cudaEventCreate(&stop));

    const int threads = 256;

    // Extended warmup (5 kernel launches)
    for (int i = 0; i < 5; ++i) {
        launch_selected_kernel(d_src, d_sink, n_float4, blocks, threads, stream, tma_ctx);
    }
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> bw_samples;
    bw_samples.reserve(trials);

    for (int t = 0; t < trials; ++t) {
        CUDA_RT_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; ++i) {
            launch_selected_kernel(d_src, d_sink, n_float4, blocks, threads, stream, tma_ctx);
        }
        CUDA_RT_CHECK(cudaEventRecord(stop, stream));
        CUDA_RT_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_RT_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double seconds = ms / 1000.0 / iters;
        double gbps = (double)buf_bytes / seconds / 1e9;
        bw_samples.push_back(gbps);
    }

    CUDA_RT_CHECK(cudaEventDestroy(start));
    CUDA_RT_CHECK(cudaEventDestroy(stop));

    return median(bw_samples);
}

// ---------------------------------------------------------------------------
// Baseline bandwidth measurement on the default/full context
// ---------------------------------------------------------------------------
static double measure_bandwidth_default(const float4 *d_src, float *d_sink,
                                        size_t n_float4, int sm_count,
                                        int iters, int trials, size_t buf_bytes,
                                        const TmaContext &tma_ctx)
{
    const int threads = 256;
    const int blocks = std::min((int)((n_float4 + threads - 1) / threads),
                                sm_count * 32);

    cudaStream_t stream;
    CUDA_RT_CHECK(cudaStreamCreate(&stream));

    double bw = measure_bandwidth_timed(stream, d_src, d_sink, n_float4,
                                        blocks, iters, trials, buf_bytes, tma_ctx);

    CUDA_RT_CHECK(cudaStreamDestroy(stream));
    return bw;
}

// ---------------------------------------------------------------------------
// Probe which SM IDs actually execute work under an externally supplied mask.
// Keep many blocks resident briefly so every enabled SM gets an opportunity.
// ---------------------------------------------------------------------------
__global__ void smid_residency_probe(unsigned int *block_counts,
                                     unsigned int block_counts_size)
{
    unsigned int smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    if (threadIdx.x == 0) {
        if (smid < block_counts_size)
            atomicAdd(&block_counts[smid], 1U);

        const unsigned long long start = clock64();
        while (clock64() - start < 100000ULL) {
            asm volatile("");
        }
    }
}

static std::vector<unsigned int> observe_sm_residency(unsigned int total_sms)
{
    constexpr unsigned int kMaxReportedSmId = 4096;
    unsigned int *d_counts = nullptr;
    CUDA_RT_CHECK(cudaMalloc(&d_counts, kMaxReportedSmId * sizeof(unsigned int)));
    CUDA_RT_CHECK(cudaMemset(d_counts, 0, kMaxReportedSmId * sizeof(unsigned int)));

    const unsigned int blocks = std::max(256U, total_sms * 32U);
    smid_residency_probe<<<blocks, 32>>>(d_counts, kMaxReportedSmId);
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> counts(kMaxReportedSmId);
    CUDA_RT_CHECK(cudaMemcpy(counts.data(), d_counts,
                             kMaxReportedSmId * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
    CUDA_RT_CHECK(cudaFree(d_counts));

    std::vector<unsigned int> observed;
    for (unsigned int smid = 0; smid < kMaxReportedSmId; ++smid) {
        if (counts[smid] != 0)
            observed.push_back(smid);
    }
    return observed;
}

// ---------------------------------------------------------------------------
// Measure bandwidth inside a Green Context with a given SM partition
// ---------------------------------------------------------------------------
static double measure_bandwidth_green_ctx(CUdevice cuDev,
                                          CUdevResource *smPartition,
                                          const float4 *d_src, float *d_sink,
                                          size_t n_float4, int iters,
                                          int trials, size_t buf_bytes,
                                          int allocated_sms,
                                          const TmaContext &tma_ctx)
{
    // 1. Generate resource descriptor
    CUdevResourceDesc desc;
    CUDA_DRIVER_CHECK(cuDevResourceGenerateDesc(&desc, smPartition, 1));

    // 2. Create Green Context
    CUgreenCtx greenCtx;
    CUDA_DRIVER_CHECK(cuGreenCtxCreate(&greenCtx, desc, cuDev,
                                       CU_GREEN_CTX_DEFAULT_STREAM));

    // 3. Convert to CUcontext and push it
    CUcontext ctx;
    CUDA_DRIVER_CHECK(cuCtxFromGreenCtx(&ctx, greenCtx));
    CUDA_DRIVER_CHECK(cuCtxPushCurrent(ctx));

    // 4. Create a stream inside the green context
    cudaStream_t stream;
    CUDA_RT_CHECK(cudaStreamCreate(&stream));

    // 5. Configure launch — enough blocks to saturate assigned SMs
    const int threads = 256;
    const int blocks = std::min((int)((n_float4 + threads - 1) / threads),
                                allocated_sms * 32);

    double bw = measure_bandwidth_timed(stream, d_src, d_sink, n_float4,
                                        blocks, iters, trials, buf_bytes, tma_ctx);

    // Cleanup
    CUDA_RT_CHECK(cudaStreamDestroy(stream));

    CUcontext popped;
    CUDA_DRIVER_CHECK(cuCtxPopCurrent(&popped));
    CUDA_DRIVER_CHECK(cuGreenCtxDestroy(greenCtx));

    return bw;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s [buffer_MB] [iterations] [gpu_id] [sm_step] [trials] [load_mode]\n", program);
    printf("  %s --external-single [buffer_MB] [iterations] [gpu_id] [trials] [load_mode]\n", program);
    printf("  %s --external-probe [gpu_id]\n", program);
    printf("\n--external-single runs one ordinary-context measurement so an external\n");
    printf("TPC-mask tool such as nvtaskset can control physical placement.\n");
    printf("--external-probe reports only SM residency and does not allocate the read buffer.\n");
}

static void print_external_usage(const char *program)
{
    printf("Usage: %s --external-single [buffer_MB] [iterations] [gpu_id] [trials] [load_mode]\n", program);
    printf("Runs one ordinary-context measurement under an externally applied TPC mask.\n");
    printf("stdout contains one external_single JSON record with bandwidth_GBps,\n");
    printf("observed_sm_count, and observed_sm_ids. Diagnostics go to stderr.\n");
}

static void print_external_probe_usage(const char *program)
{
    printf("Usage: %s --external-probe [gpu_id]\n", program);
    printf("Runs only the SM-residency probe under an externally applied TPC mask.\n");
    printf("stdout contains one external_probe JSON record with total_sm_count,\n");
    printf("observed_sm_count, and observed_sm_ids. Diagnostics go to stderr.\n");
}

int main(int argc, char **argv)
{
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    const bool external_single = argc > 1 && strcmp(argv[1], "--external-single") == 0;
    const bool external_probe = argc > 1 && strcmp(argv[1], "--external-probe") == 0;
    if (external_single && argc > 2 &&
        (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0)) {
        print_external_usage(argv[0]);
        return 0;
    }
    if (external_probe && argc > 2 &&
        (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0)) {
        print_external_probe_usage(argv[0]);
        return 0;
    }

    // ---- Parse arguments ----
    size_t buf_mb;
    int iters, dev_id, sm_step, trials, mode;
    if (external_probe) {
        buf_mb = 0;
        iters = 1;
        dev_id = (argc > 2) ? atoi(argv[2]) : 0;
        trials = 1;
        mode = 1;
        sm_step = 0;
    } else if (external_single) {
        buf_mb = (argc > 2) ? atol(argv[2]) : 512;
        iters  = (argc > 3) ? atoi(argv[3]) : 20;
        dev_id = (argc > 4) ? atoi(argv[4]) : 0;
        trials = (argc > 5) ? atoi(argv[5]) : 5;
        mode   = (argc > 6) ? atoi(argv[6]) : 1;
        sm_step = 0;
    } else {
        buf_mb = (argc > 1) ? atol(argv[1]) : 512;
        iters  = (argc > 2) ? atoi(argv[2]) : 20;
        dev_id = (argc > 3) ? atoi(argv[3]) : 0;
        sm_step = (argc > 4) ? atoi(argv[4]) : 0;
        trials = (argc > 5) ? atoi(argv[5]) : 5;
        mode   = (argc > 6) ? atoi(argv[6]) : 0;
    }
    if (!external_probe && (buf_mb == 0 || iters <= 0 || trials <= 0)) {
        fprintf(stderr, "buffer_MB, iterations, and trials must be positive\n");
        return EXIT_FAILURE;
    }
    g_load_mode = static_cast<LoadMode>(std::max(0, std::min(mode, 5)));

    // ---- Initialize CUDA Driver API ----
    CUDA_DRIVER_CHECK(cuInit(0));

    CUdevice cuDev;
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDev, dev_id));

    char devName[256];
    CUDA_DRIVER_CHECK(cuDeviceGetName(devName, sizeof(devName), cuDev));

    int cc_major, cc_minor;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&cc_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDev));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&cc_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDev));

    // ---- Ensure primary context is active (required for green contexts) ----
    CUDA_RT_CHECK(cudaSetDevice(dev_id));
    CUDA_RT_CHECK(cudaFree(nullptr));  // force primary context init

    // ---- Query total SM count ----
    CUdevResource totalSmRes;
    CUDA_DRIVER_CHECK(cuDeviceGetDevResource(cuDev, &totalSmRes,
                                              CU_DEV_RESOURCE_TYPE_SM));
    unsigned int totalSMs = totalSmRes.sm.smCount;

    const char *benchmark_name = external_probe
        ? "External-mask SM Residency Probe"
        : (external_single
            ? "External-mask DRAM Bandwidth Measurement"
            : "Green Context Bandwidth Saturation Benchmark");
    fprintf(stderr, "=== %s ===\n", benchmark_name);
    fprintf(stderr, "GPU:             %s\n", devName);
    fprintf(stderr, "Compute Cap:     %d.%d\n", cc_major, cc_minor);
    fprintf(stderr, "Total SMs:       %u\n", totalSMs);
    if (!external_probe) {
        fprintf(stderr, "Buffer Size:     %zu MB\n", buf_mb);
        fprintf(stderr, "Iterations:      %d (per trial)\n", iters);
        fprintf(stderr, "Trials:          %d (median)\n", trials);
        fprintf(stderr, "Load Mode:       %s\n", load_mode_name(g_load_mode));
    }

    if (external_probe) {
        const std::vector<unsigned int> observed_sms = observe_sm_residency(totalSMs);
        if (observed_sms.empty()) {
            fprintf(stderr, "SM-residency probe observed no active SMs\n");
            return EXIT_FAILURE;
        }
        fprintf(stderr, "Observed SMs:    %zu\n", observed_sms.size());
        printf("{\"kind\":\"external_probe\",\"total_sm_count\":%u,"
               "\"observed_sm_count\":%zu,\"observed_sm_ids\":[",
               totalSMs, observed_sms.size());
        for (size_t i = 0; i < observed_sms.size(); ++i) {
            if (i != 0)
                printf(",");
            printf("%u", observed_sms[i]);
        }
        printf("]}\n");
        fflush(stdout);
        return 0;
    }

    // Determine SM granularity for this architecture
    // Blackwell (10.x) / Hopper (9.x): 8 SM granularity
    // Ampere (8.x): 4 SMs, multiple of 2
    // Volta/Turing (7.x): 2 SMs, multiple of 2
    int sm_granularity;
    if (cc_major >= 9)
        sm_granularity = 8;
    else if (cc_major == 8)
        sm_granularity = 4;   // minimum, must be multiple of 2
    else if (cc_major == 7)
        sm_granularity = 2;
    else
        sm_granularity = 1;

    if (sm_step == 0)
        sm_step = sm_granularity;
    else
        sm_step = std::max(sm_step, sm_granularity);

    fprintf(stderr, "SM Granularity:  %d\n", sm_granularity);
    fprintf(stderr, "SM Step:         %d\n", sm_step);
    fprintf(stderr, "\n");

    // ---- Allocate device memory ----
    size_t buf_bytes = buf_mb * 1024ULL * 1024ULL;
    size_t n_float4 = buf_bytes / sizeof(float4);
    buf_bytes = n_float4 * sizeof(float4);  // align

    float4 *d_src = nullptr;
    float  *d_sink = nullptr;
    CUDA_RT_CHECK(cudaMalloc(&d_src, buf_bytes));
    // Sink buffer — one float per possible thread
    size_t sink_size = (size_t)totalSMs * 32 * 256 * sizeof(float);
    CUDA_RT_CHECK(cudaMalloc(&d_sink, sink_size));
    CUDA_RT_CHECK(cudaMemset(d_src, 0x42, buf_bytes));
    CUDA_RT_CHECK(cudaMemset(d_sink, 0, sink_size));

    TmaContext tma_ctx{};
    if (uses_tma(g_load_mode)) {
        if (cc_major < 9) {
            fprintf(stderr, "TMA mode requires SM90+; this GPU is %d.%d\n", cc_major, cc_minor);
            return EXIT_FAILURE;
        }
        int max_optin_smem = 0;
        CUDA_RT_CHECK(cudaDeviceGetAttribute(
            &max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
        if (kTmaDynamicSmemBytes > static_cast<size_t>(max_optin_smem)) {
            fprintf(stderr,
                    "TMA mode requires %zu bytes of dynamic shared memory, "
                    "but this GPU supports at most %d bytes per block\n",
                    kTmaDynamicSmemBytes, max_optin_smem);
            return EXIT_FAILURE;
        }
        CUDA_RT_CHECK(cudaFuncSetAttribute(
            read_bandwidth_kernel_tma,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kTmaDynamicSmemBytes)));
        fprintf(stderr, "TMA Stage:       %zu KiB x %zu buffers (%zu KiB dynamic shared)\n",
                kTmaStageBytes / 1024, kTmaStages, kTmaDynamicSmemBytes / 1024);
        if (!build_tma_context(cuDev, d_src, buf_bytes, &tma_ctx)) {
            fprintf(stderr, "Failed to build TMA tensor map\n");
            return EXIT_FAILURE;
        }
    }

    if (external_single) {
        const double bw = measure_bandwidth_default(d_src, d_sink, n_float4,
                                                     totalSMs, iters, trials,
                                                     buf_bytes, tma_ctx);
        const std::vector<unsigned int> observed_sms = observe_sm_residency(totalSMs);
        if (observed_sms.empty()) {
            fprintf(stderr, "SM-residency probe observed no active SMs\n");
            return EXIT_FAILURE;
        }

        fprintf(stderr, "Observed SMs:    %zu\n", observed_sms.size());
        fprintf(stderr, "Bandwidth:       %.3f GB/s\n", bw);
        printf("{\"kind\":\"external_single\",\"bandwidth_GBps\":%.6f,"
               "\"observed_sm_count\":%zu,\"observed_sm_ids\":[",
               bw, observed_sms.size());
        for (size_t i = 0; i < observed_sms.size(); ++i) {
            if (i != 0)
                printf(",");
            printf("%u", observed_sms[i]);
        }
        printf("]}\n");
        fflush(stdout);

        if (tma_ctx.enabled && tma_ctx.d_tensor_map != nullptr)
            CUDA_RT_CHECK(cudaFree(tma_ctx.d_tensor_map));
        CUDA_RT_CHECK(cudaFree(d_src));
        CUDA_RT_CHECK(cudaFree(d_sink));
        return 0;
    }

    // ---- Baseline: full-GPU bandwidth ----
    double bw_full = measure_bandwidth_default(d_src, d_sink, n_float4,
                                                totalSMs, iters, trials,
                                                buf_bytes, tma_ctx);
    fprintf(stderr, "Full-GPU bandwidth (median of %d trials): %.2f GB/s\n\n",
            trials, bw_full);

    // ---- CSV header ----
    printf("sm_count_requested,sm_count_allocated,bandwidth_GBps,pct_of_full_gpu\n");
    printf("%u,%u,%.3f,%.1f\n", totalSMs, totalSMs, bw_full, 100.0);

    // ---- Sweep SM counts ----
    std::vector<unsigned int> sm_counts;
    for (unsigned int s = sm_granularity; s < totalSMs; s += sm_step)
        sm_counts.push_back(s);

    for (unsigned int requested_sms : sm_counts) {
        // Query how many groups we'd get with this minCount
        unsigned int nbGroups = 0;
        CUdevResource remaining;
        CUDA_DRIVER_CHECK(cuDevSmResourceSplitByCount(
            nullptr, &nbGroups, &totalSmRes, &remaining, 0, requested_sms));

        if (nbGroups == 0) {
            fprintf(stderr, "  SM=%u: cannot create partition, skipping\n",
                    requested_sms);
            continue;
        }

        // Actually split — we only need 1 group
        unsigned int oneGroup = 1;
        CUdevResource partition;
        CUDA_DRIVER_CHECK(cuDevSmResourceSplitByCount(
            &partition, &oneGroup, &totalSmRes, &remaining, 0, requested_sms));

        unsigned int allocated = partition.sm.smCount;

        double bw = measure_bandwidth_green_ctx(cuDev, &partition,
                                                 d_src, d_sink, n_float4,
                                                 iters, trials, buf_bytes,
                                                 allocated, tma_ctx);

        double pct = (bw / bw_full) * 100.0;

        printf("%u,%u,%.3f,%.1f\n", requested_sms, allocated, bw, pct);
        fflush(stdout);

        fprintf(stderr, "  SM=%u (alloc=%u): %.2f GB/s (%.1f%%)\n",
                requested_sms, allocated, bw, pct);
    }

    // ---- Summary ----
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Full-GPU BW (baseline): %.2f GB/s\n", bw_full);

    // ---- Cleanup ----
    if (tma_ctx.enabled && tma_ctx.d_tensor_map != nullptr) {
        CUDA_RT_CHECK(cudaFree(tma_ctx.d_tensor_map));
    }
    CUDA_RT_CHECK(cudaFree(d_src));
    CUDA_RT_CHECK(cudaFree(d_sink));

    return 0;
}
