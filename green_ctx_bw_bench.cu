/**
 * green_ctx_bw_bench.cu
 *
 * Microbenchmark: Effective Read Bandwidth vs. SM Allocation
 * using CUDA Green Contexts on NVIDIA Blackwell (B200/GB200).
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
 *   ./green_ctx_bw_bench [buffer_size_MB] [iterations] [gpu_id] [sm_step] [trials] [load_mode]
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
#include <cmath>
#include <random>
#include <string>
#include "bench_options.h"
 
enum class LoadMode {
    Default = 0,
    L1Bypass = 1,
    CpAsync = 2,
    PipelineAsync = 3,
    Tma = 4,
    DecodeLike = 5,
};

static constexpr size_t kTmaTileBytes = 16384;
static constexpr size_t kTmaRowBytes = 256;
static constexpr size_t kTmaRowsPerTile = kTmaTileBytes / kTmaRowBytes;
static constexpr size_t kTmaTileFloat4s = kTmaTileBytes / sizeof(float4);
static constexpr size_t kDecodeTileBytes = 4096;
static constexpr size_t kDecodeTileFloat4s = kDecodeTileBytes / sizeof(float4);
static constexpr size_t kDecodeKvFloat4s = kDecodeTileFloat4s / 2;

static LoadMode g_load_mode = LoadMode::Default;
static BenchOptions g_options;

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
// L1-bypass variant: ld.global.cg still caches in L2.
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
        :: "r"(smem), "l"(gmem_ptr), "n"(16) : "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait0()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
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

    for (size_t base = blockIdx.x * (size_t)blockDim.x; base < n_float4; base += stride) {
        size_t i = base + threadIdx.x;
        if (i < n_float4) cp_async_16B(&smem[threadIdx.x], &src[i]);
        cp_async_commit();
        cp_async_wait0();
        if (i < n_float4) {
            float4 v = smem[threadIdx.x];
            accum.x += v.x; accum.y += v.y;
            accum.z += v.z; accum.w += v.w;
        }
        __syncthreads();  // Finish reads before reusing shared memory.
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

    for (size_t base = blockIdx.x * (size_t)blockDim.x; base < n_float4; base += stride) {
        // Every member supplies the same pointers and the collective byte count.
        size_t count = min((size_t)blockDim.x, n_float4 - base);
        cg::memcpy_async(block, smem, src + base, count * sizeof(float4));
        cg::wait(block);
        if (threadIdx.x < count) {
            float4 v = smem[threadIdx.x];
            accum.x += v.x; accum.y += v.y;
            accum.z += v.z; accum.w += v.w;
        }
        block.sync();  // All readers finish before the next collective overwrites smem.
    }

    if (tid < stride)
        sink[tid] = accum.x + accum.y + accum.z + accum.w;
}

template<int Stages>
__global__ void __launch_bounds__(256)
read_bandwidth_kernel_tma(const float4 *__restrict__ src,
                          float *__restrict__ sink,
                          size_t n_float4,
                          const CUtensorMap *tensor_map,
                          bool consume)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) unsigned char smem_tiles[];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bars[Stages];
    if (threadIdx.x == 0) {
        for (int s = 0; s < Stages; ++s) init(&bars[s], 1);
        // Publish initialization from the generic proxy to the TMA async proxy.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    const size_t tile_count = (n_float4 + kTmaTileFloat4s - 1) / kTmaTileFloat4s;
    auto issue_tile = [&](int stage, size_t tile) {
        (void)cuda::device::barrier_arrive_tx(bars[stage], 1, kTmaTileBytes);
        uint32_t dst = __cvta_generic_to_shared(smem_tiles + stage * kTmaTileBytes);
        uint32_t bar = __cvta_generic_to_shared(cuda::device::barrier_native_handle(bars[stage]));
        int row = static_cast<int>(tile * kTmaRowsPerTile);
        // Stable PTX interface, independent of deprecated experimental C++ wrappers.
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
            "[%0], [%1, {0, %2}], [%3];"
            :: "r"(dst), "l"(tensor_map), "r"(row), "r"(bar) : "memory");
    };
    if (threadIdx.x == 0) {
        for (int s = 0; s < Stages; ++s) {
            size_t tile = blockIdx.x + (size_t)s * gridDim.x;
            if (tile < tile_count) issue_tile(s, tile);
        }
    }

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
    size_t iteration = 0;
    for (size_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x, ++iteration) {
        int stage = iteration % Stages;
        // Each reuse advances this barrier by one phase; no arrival-token array needed.
        if (threadIdx.x == 0) bars[stage].wait_parity((iteration / Stages) & 1);
        __syncthreads();
        // Runtime-uniform branch: both controls use the SAME compiled kernel,
        // hence the same registers, shared memory and theoretical occupancy.
        if (consume) {
            const float4 *data = reinterpret_cast<const float4 *>(smem_tiles + stage * kTmaTileBytes);
            for (size_t i = threadIdx.x; i < kTmaTileFloat4s; i += blockDim.x) {
                if (tile * kTmaTileFloat4s + i < n_float4) {
                    float4 v = data[i];
                    accum.x += v.x; accum.y += v.y;
                    accum.z += v.z; accum.w += v.w;
                }
            }
        }
        __syncthreads();  // Protect buffer reuse until all consumers finish reading.
        size_t future_tile = tile + (size_t)Stages * gridDim.x;
        if (threadIdx.x == 0 && future_tile < tile_count) issue_tile(stage, future_tile);
    }
    // Copy-only still waits for EVERY transfer, keeps both CTA barriers, and
    // writes the completed-tile count to the same per-thread sink footprint.
    // The volatile TMA asm above cannot be discarded with the consumer absent.
    sink[(size_t)blockIdx.x * blockDim.x + threadIdx.x] =
        consume ? accum.x + accum.y + accum.z + accum.w : static_cast<float>(iteration);
#else
    // Never report a bandwidth for a kernel compiled without TMA support.
    __trap();
#endif
}

__global__ void __launch_bounds__(256)
read_bandwidth_kernel_decode_like(const float4 *__restrict__ src,
                                  float *__restrict__ sink,
                                  size_t n_float4)
{
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t token_count = (n_float4 + kDecodeTileFloat4s - 1) / kDecodeTileFloat4s;

    float accum = 0.0f;
    const float q_scale = 1.0f + 0.01f * static_cast<float>(blockIdx.x & 31);

    for (size_t token = blockIdx.x; token < token_count; token += gridDim.x) {
        const size_t token_base = token * kDecodeTileFloat4s;
        float token_accum = 0.0f;

        // Lightweight attention-like KV-cache sweep: read K and V halves,
        // apply a tiny per-block scale, and keep the math cheap so bandwidth
        // remains the bottleneck.
        for (size_t i = threadIdx.x; i < kDecodeKvFloat4s; i += blockDim.x) {
            float4 k = make_float4(0, 0, 0, 0), v = k;
            if (token_base + i < n_float4) k = __ldg(&src[token_base + i]);
            if (token_base + kDecodeKvFloat4s + i < n_float4)
                v = __ldg(&src[token_base + kDecodeKvFloat4s + i]);
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

    if (buf_bytes == 0 || buf_bytes % kTmaRowBytes ||
        buf_bytes / kTmaRowBytes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "TMA buffer must contain whole 256-byte rows and fit signed coordinates\n");
        exit(EXIT_FAILURE);
    }
    CUtensorMap host_map{};
    const cuuint64_t global_dim[2]     = { kTmaRowBytes, buf_bytes / kTmaRowBytes };
    const cuuint64_t global_strides[1] = { kTmaRowBytes };            // dim1 간 stride = 256 bytes
    const cuuint32_t box_dim[2]        = { kTmaRowBytes, kTmaRowsPerTile };  // {256, 64}
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
                                   const TmaContext &tma_ctx,
                                   bool check_tma_payload = false)
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
        switch (g_options.tma_stages) {
        case 2: read_bandwidth_kernel_tma<2><<<blocks, threads, 2 * kTmaTileBytes, stream>>>(d_src, d_sink, n_float4, tma_ctx.d_tensor_map, check_tma_payload || !g_options.tma_copy_only); break;
        case 4: read_bandwidth_kernel_tma<4><<<blocks, threads, 4 * kTmaTileBytes, stream>>>(d_src, d_sink, n_float4, tma_ctx.d_tensor_map, check_tma_payload || !g_options.tma_copy_only); break;
        case 8: read_bandwidth_kernel_tma<8><<<blocks, threads, 8 * kTmaTileBytes, stream>>>(d_src, d_sink, n_float4, tma_ctx.d_tensor_map, check_tma_payload || !g_options.tma_copy_only); break;
        }
        break;
    case LoadMode::DecodeLike:
        read_bandwidth_kernel_decode_like<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
        break;
    }
    CUDA_RT_CHECK(cudaGetLastError());
}

static const void *selected_kernel()
{
    switch (g_load_mode) {
    case LoadMode::Default: return reinterpret_cast<const void *>(read_bandwidth_kernel);
    case LoadMode::L1Bypass: return reinterpret_cast<const void *>(read_bandwidth_kernel_nc);
    case LoadMode::CpAsync: return reinterpret_cast<const void *>(read_bandwidth_kernel_cp_async);
    case LoadMode::PipelineAsync: return reinterpret_cast<const void *>(read_bandwidth_kernel_pipeline);
    case LoadMode::DecodeLike: return reinterpret_cast<const void *>(read_bandwidth_kernel_decode_like);
    case LoadMode::Tma:
        if (g_options.tma_stages == 4) return reinterpret_cast<const void *>(read_bandwidth_kernel_tma<4>);
        if (g_options.tma_stages == 8) return reinterpret_cast<const void *>(read_bandwidth_kernel_tma<8>);
        return reinterpret_cast<const void *>(read_bandwidth_kernel_tma<2>);
    }
    return nullptr;
}

static void configure_selected_kernel(int cc_major, int dev_id)
{
    cudaFuncAttributes attr{};
    CUDA_RT_CHECK(cudaFuncGetAttributes(&attr, selected_kernel()));
    const int minimum = uses_tma(g_load_mode) ? 90 :
        (g_load_mode == LoadMode::CpAsync ? 80 : 0);
    if (cc_major * 10 < minimum || attr.ptxVersion < minimum) {
        fprintf(stderr, "Selected load mode requires both GPU and compiled PTX >= sm_%d; "
                "GPU major=%d, compiled PTX=%d. Rebuild for this GPU.\n",
                minimum, cc_major, attr.ptxVersion);
        exit(EXIT_FAILURE);
    }
    size_t dynamic_bytes = uses_tma(g_load_mode)
        ? g_options.tma_stages * kTmaTileBytes :
          (uses_shared_staging(g_load_mode) ? 256 * sizeof(float4) : 0);
    if (uses_tma(g_load_mode)) {
        int maximum = 0;
        CUDA_RT_CHECK(cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
        if (dynamic_bytes + attr.sharedSizeBytes > static_cast<size_t>(maximum)) {
            fprintf(stderr, "TMA stages require %zu shared bytes, GPU limit is %d\n",
                    dynamic_bytes + attr.sharedSizeBytes, maximum);
            exit(EXIT_FAILURE);
        }
        CUDA_RT_CHECK(cudaFuncSetAttribute(selected_kernel(),
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_bytes)));
    }
    int resident_blocks = 0;
    CUDA_RT_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, selected_kernel(), 256, dynamic_bytes));
    fprintf(stderr, "Kernel configuration: mode=%d PTX=sm_%d registers=%d "
            "shared_bytes=%zu resident_blocks_per_sm=%d blocks_per_sm=%d tma_stages=%d tma_copy_only=%d\n",
            static_cast<int>(g_load_mode), attr.ptxVersion, attr.numRegs,
            dynamic_bytes + attr.sharedSizeBytes, resident_blocks,
            g_options.blocks_per_sm, g_options.tma_stages, g_options.tma_copy_only);
}

static int grid_blocks(size_t n_float4, int sms)
{
    size_t work = uses_tma(g_load_mode)
        ? (n_float4 + kTmaTileFloat4s - 1) / kTmaTileFloat4s
        : (n_float4 + 255) / 256;
    return static_cast<int>(std::min(work, static_cast<size_t>(sms) * g_options.blocks_per_sm));
}

// Small, varied input with exact integer sums. CPU ownership is derived from
// global indices, independently of the device kernels' grid-stride loops.
static float validation_value(size_t index, unsigned lane)
{
    uint32_t x = static_cast<uint32_t>(index * 4 + lane) * 747796405U + 2891336453U;
    x ^= x >> 16;
    return static_cast<float>(static_cast<int>(x & 15U) - 7);
}

struct ValidationStats { size_t cases = 0, checked_float4s = 0, completion_cases = 0; };

static ValidationStats validate_selected_kernel(CUdevice device, cudaStream_t stream)
{
    ValidationStats stats;
    const size_t sizes[] = {256, 16128, 16384, 16640, 65792, 1048832};
    const int grids[] = {1, 3, 17};
    for (size_t bytes : sizes) {
        const size_t count = bytes / sizeof(float4);
        std::vector<float4> input(count);
        for (size_t i = 0; i < count; ++i)
            input[i] = make_float4(validation_value(i, 0), validation_value(i, 1),
                                   validation_value(i, 2), validation_value(i, 3));
        float4 *src = nullptr;
        float *sink = nullptr;
        CUDA_RT_CHECK(cudaMalloc(&src, bytes));
        CUDA_RT_CHECK(cudaMalloc(&sink, 17 * 256 * sizeof(float)));
        CUDA_RT_CHECK(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));
        TmaContext tma{};
        if (uses_tma(g_load_mode)) build_tma_context(device, src, bytes, &tma);
        for (int blocks : grids) {
            std::vector<double> expected(blocks * 256, 0.0);
            for (size_t i = 0; i < count; ++i) {
                size_t owner = i % expected.size();
                double weight = 1.0;
                if (uses_tma(g_load_mode))
                    owner = ((i / kTmaTileFloat4s) % blocks) * 256 + (i % kTmaTileFloat4s) % 256;
                if (g_load_mode == LoadMode::DecodeLike) {
                    size_t block = (i / kDecodeTileFloat4s) % blocks;
                    size_t offset = i % kDecodeTileFloat4s;
                    owner = block * 256 + offset % kDecodeKvFloat4s;
                    weight = offset < kDecodeKvFloat4s ?
                        static_cast<double>(1.0f + 0.01f * static_cast<float>(block & 31)) : 0.125;
                }
                const float4 v = input[i];
                expected[owner] += (static_cast<double>(v.x) + v.y + v.z + v.w) * weight;
            }
            CUDA_RT_CHECK(cudaMemsetAsync(sink, 0xff, expected.size() * sizeof(float), stream));
            // Validate payload with the consume branch of the same kernel;
            // these shared-memory reads are OUTSIDE benchmark timing.
            launch_selected_kernel(src, sink, count, blocks, 256, stream, tma, true);
            CUDA_RT_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> actual(expected.size());
            CUDA_RT_CHECK(cudaMemcpy(actual.data(), sink, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) ||
                    std::abs(actual[i] - expected[i]) > 0.002 + 0.0002 * std::abs(expected[i])) {
                    fprintf(stderr, "Validation FAILED mode=%d stages=%d bytes=%zu blocks=%d "
                            "sink=%zu expected=%.9g actual=%.9g\n",
                            static_cast<int>(g_load_mode), g_options.tma_stages,
                            bytes, blocks, i, expected[i], actual[i]);
                    exit(EXIT_FAILURE);
                }
            }
            ++stats.cases;
            stats.checked_float4s += count;
            if (g_options.tma_copy_only) {
                // Exercise the actual no-consumer branch too, including idle
                // CTAs, partial tiles and repeated barrier phases.
                CUDA_RT_CHECK(cudaMemsetAsync(sink, 0xff, actual.size() * sizeof(float), stream));
                launch_selected_kernel(src, sink, count, blocks, 256, stream, tma);
                CUDA_RT_CHECK(cudaStreamSynchronize(stream));
                CUDA_RT_CHECK(cudaMemcpy(actual.data(), sink, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                const size_t tiles = (count + kTmaTileFloat4s - 1) / kTmaTileFloat4s;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const size_t block = i / 256;
                    const float completed = static_cast<float>(tiles / blocks + (block < tiles % blocks));
                    if (actual[i] != completed) {
                        fprintf(stderr, "Copy-only completion FAILED stages=%d bytes=%zu blocks=%d "
                                "sink=%zu expected=%g actual=%g\n",
                                g_options.tma_stages, bytes, blocks, i, completed, actual[i]);
                        exit(EXIT_FAILURE);
                    }
                }
                ++stats.completion_cases;
            }
        }
        if (tma.enabled) CUDA_RT_CHECK(cudaFree(tma.d_tensor_map));
        CUDA_RT_CHECK(cudaFree(src));
        CUDA_RT_CHECK(cudaFree(sink));
    }
    fprintf(stderr, "Validation PASSED: %zu cases, %zu float4 inputs checked\n",
            stats.cases, stats.checked_float4s);
    if (g_options.tma_copy_only)
        fprintf(stderr, "Copy-only completion checks PASSED: %zu cases; payload checked separately outside timing\n",
                stats.completion_cases);
    return stats;
}

__global__ void initialize_benchmark_input(float4 *src, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += (size_t)blockDim.x * gridDim.x) {
        uint32_t x = static_cast<uint32_t>(i) * 747796405U + 2891336453U;
        x ^= x >> 16;
        // Normal finite floats with varied mantissas, avoiding uniform data.
        src[i] = make_float4(__uint_as_float(0x3f000000U | (x & 0x007fffffU)),
                            __uint_as_float(0x3f800000U | ((x * 1664525U) & 0x007fffffU)),
                            __uint_as_float(0xbf000000U | ((x * 1013904223U) & 0x007fffffU)),
                            __uint_as_float(0xbf800000U | ((x * 22695477U) & 0x007fffffU)));
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
        if (!(ms > 0) || !std::isfinite(ms)) {
            fprintf(stderr, "Invalid CUDA event duration: %g ms\n", ms);
            exit(EXIT_FAILURE);
        }
        double seconds = ms / 1000.0 / iters;
        double gbps = (double)buf_bytes / seconds / 1e9;
        bw_samples.push_back(gbps);
        fprintf(stderr, "trial=%d blocks=%d kernel_ms=%.6f bandwidth_GBps=%.6f\n",
                t, blocks, ms / iters, gbps);
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
    const int blocks = grid_blocks(n_float4, sm_count);

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
static unsigned int g_probe_observed = 0;

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

    if (!g_options.skip_validation) validate_selected_kernel(cuDev, stream);
    auto observed = observe_sm_residency(allocated_sms);
    g_probe_observed = observed.size();
    if (g_probe_observed > static_cast<unsigned int>(allocated_sms)) {
        fprintf(stderr, "SM residency exceeded allocation: observed=%u allocated=%d\n",
                g_probe_observed, allocated_sms);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Green partition: allocated_sms=%d probe_observed_sms=%u\n",
            allocated_sms, g_probe_observed);
    fprintf(stderr, "Probe SM IDs:");
    for (unsigned int id : observed) fprintf(stderr, " %u", id);
    fprintf(stderr, "\n");

    // 5. Configure launch — enough blocks to saturate assigned SMs
    const int blocks = grid_blocks(n_float4, allocated_sms);

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
    printf("Usage:\n  %s [buffer_MB] [iterations] [gpu_id] [sm_step] [trials] [load_mode]\n", program);
    printf("  %s --external-single [buffer_MB] [iterations] [gpu_id] [trials] [load_mode]\n", program);
    printf("  %s --external-probe [gpu_id]\n", program);
    printf("Options (may precede or follow positional arguments):\n");
    printf("  --validate-only     Check all output threads on varied input, then exit\n");
    printf("  --skip-validation   Skip correctness checks (profiling after validation only)\n");
    printf("  --blocks-per-sm N   Queued CTAs per allocated SM, 1..256 (default 32)\n");
    printf("  --tma-stages N      TMA pipeline stages: 2, 4, or 8 (default 2)\n");
    printf("  --tma-copy-only     TMA transfer/wait control without tile reads or FP32 sums (mode 4)\n");
    printf("  --sm-count N        Measure one partition, emitting a single JSON record\n");
    printf("  --require-dram      Require input >= 4 times the GPU's L2 capacity\n");
    printf("  --seed N            Randomized sweep order (default 8228)\n");
    printf("Modes: 0=__ldg, 1=ld.global.cg, 2=cp.async, 3=cooperative copy, 4=TMA, 5=decode-like\n");
    printf("Correctness checks run before timing unless --skip-validation is supplied.\n");
    printf("--external-single emits one external_single JSON record with bandwidth_GBps,\n");
    printf("observed_sm_count and observed_sm_ids; --external-probe emits external_probe JSON\n");
    printf("with residency only. These modes support external nvtaskset placement.\n");
    printf("B200 native builds require CUDA >= 12.8 and CUDA_ARCH=sm_100.\n");
}

static void print_sm_ids(const std::vector<unsigned int> &ids)
{
    for (size_t i = 0; i < ids.size(); ++i) printf("%s%u", i ? "," : "", ids[i]);
}

int main(int argc, char **argv)
{
    try {
        g_options = parse_options(argc, argv);
    } catch (const std::exception &e) {
        fprintf(stderr, "Argument error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    if (g_options.help) { print_usage(argv[0]); return 0; }
    g_load_mode = static_cast<LoadMode>(g_options.load_mode);
    const int dev_id = g_options.gpu_id;
    const int iters = g_options.iterations, trials = g_options.trials;
    const size_t buf_bytes = g_options.buffer_mb * 1024ULL * 1024ULL;
    const size_t n_float4 = buf_bytes / sizeof(float4);

    CUDA_DRIVER_CHECK(cuInit(0));
    CUdevice cuDev;
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDev, dev_id));
    char devName[256];
    CUDA_DRIVER_CHECK(cuDeviceGetName(devName, sizeof(devName), cuDev));
    int cc_major = 0, cc_minor = 0;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDev));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDev));
    CUDA_RT_CHECK(cudaSetDevice(dev_id));
    CUDA_RT_CHECK(cudaFree(nullptr));
    CUdevResource totalSmRes;
    CUDA_DRIVER_CHECK(cuDeviceGetDevResource(cuDev, &totalSmRes, CU_DEV_RESOURCE_TYPE_SM));
    const unsigned int totalSMs = totalSmRes.sm.smCount;
    if (g_options.sm_count > static_cast<int>(totalSMs)) {
        fprintf(stderr, "--sm-count exceeds the GPU's %u SMs\n", totalSMs);
        return EXIT_FAILURE;
    }
    fprintf(stderr, "=== Green Context Bandwidth Benchmark ===\nGPU: %s\n"
            "Compute capability: %d.%d\nTotal SMs: %u\nLoad mode: %s\n",
            devName, cc_major, cc_minor, totalSMs, load_mode_name(g_load_mode));

    if (g_options.external_probe) {
        auto ids = observe_sm_residency(totalSMs);
        if (ids.empty()) { fprintf(stderr, "Residency probe observed no SMs\n"); return EXIT_FAILURE; }
        printf("{\"kind\":\"external_probe\",\"total_sm_count\":%u,\"observed_sm_count\":%zu,\"observed_sm_ids\":[", totalSMs, ids.size());
        print_sm_ids(ids); printf("]}\n");
        return 0;
    }
    configure_selected_kernel(cc_major, dev_id);
    if (!g_options.skip_validation || g_options.validate_only) {
        cudaStream_t stream;
        CUDA_RT_CHECK(cudaStreamCreate(&stream));
        auto stats = validate_selected_kernel(cuDev, stream);
        CUDA_RT_CHECK(cudaStreamDestroy(stream));
        if (g_options.validate_only) {
            printf("{\"kind\":\"validation\",\"cases\":%zu,\"checked_float4s\":%zu,\"load_mode\":%d,"
                   "\"tma_stages\":%d,\"tma_copy_only\":%s,\"completion_cases\":%zu}\n",
                    stats.cases, stats.checked_float4s, g_options.load_mode, g_options.tma_stages,
                    g_options.tma_copy_only ? "true" : "false", stats.completion_cases);
            return 0;
        }
    }
    int l2_bytes = 0;
    CUDA_RT_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev_id));
    if (buf_bytes < static_cast<size_t>(l2_bytes) * 4) {
        fprintf(stderr, "Input is below 4x L2 capacity; measured bandwidth may include cache hits.\n");
        if (g_options.require_dram) return EXIT_FAILURE;
    }
    fprintf(stderr, "Buffer: %zu MiB; L2: %d bytes; iterations=%d trials=%d seed=%d\n",
            g_options.buffer_mb, l2_bytes, iters, trials, g_options.seed);
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_RT_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t sink_size = static_cast<size_t>(totalSMs) * g_options.blocks_per_sm * 256 * sizeof(float);
    if (buf_bytes > free_bytes || sink_size > free_bytes - buf_bytes) {
        fprintf(stderr, "Requested buffers do not fit free GPU memory (%zu bytes)\n", free_bytes);
        return EXIT_FAILURE;
    }
    float4 *d_src = nullptr;
    float *d_sink = nullptr;
    CUDA_RT_CHECK(cudaMalloc(&d_src, buf_bytes));
    CUDA_RT_CHECK(cudaMalloc(&d_sink, sink_size));
    initialize_benchmark_input<<<totalSMs * 8, 256>>>(d_src, n_float4);
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaDeviceSynchronize());
    TmaContext tma{};
    if (uses_tma(g_load_mode)) build_tma_context(cuDev, d_src, buf_bytes, &tma);
    auto ids = observe_sm_residency(totalSMs);
    if (ids.empty()) { fprintf(stderr, "Residency probe observed no SMs\n"); return EXIT_FAILURE; }

    auto measure_partition = [&](unsigned int requested, unsigned int *allocated) {
        CUdevResource partition, remaining;
        unsigned int one_group = 1;
        CUDA_DRIVER_CHECK(cuDevSmResourceSplitByCount(
            &partition, &one_group, &totalSmRes, &remaining, 0, requested));
        if (one_group != 1) { fprintf(stderr, "Cannot allocate requested partition\n"); exit(EXIT_FAILURE); }
        *allocated = partition.sm.smCount;
        return measure_bandwidth_green_ctx(cuDev, &partition, d_src, d_sink,
            n_float4, iters, trials, buf_bytes, *allocated, tma);
    };
    if (g_options.sm_count) {
        unsigned int allocated = g_options.sm_count;
        g_probe_observed = ids.size();
        double bw = allocated == totalSMs
            ? measure_bandwidth_default(d_src, d_sink, n_float4, totalSMs, iters, trials, buf_bytes, tma)
            : measure_partition(g_options.sm_count, &allocated);
        printf("{\"kind\":\"single\",\"sm_count_requested\":%d,\"sm_count_allocated\":%u,"
               "\"probe_observed_sm_count\":%u,\"bandwidth_GBps\":%.6f,\"tma_copy_only\":%s}\n",
                g_options.sm_count, allocated, g_probe_observed, bw,
                g_options.tma_copy_only ? "true" : "false");
    } else if (g_options.external_single) {
        double bw = measure_bandwidth_default(d_src, d_sink, n_float4, totalSMs, iters, trials, buf_bytes, tma);
        printf("{\"kind\":\"external_single\",\"bandwidth_GBps\":%.6f,\"tma_copy_only\":%s,"
               "\"observed_sm_count\":%zu,\"observed_sm_ids\":[",
               bw, g_options.tma_copy_only ? "true" : "false", ids.size());
        print_sm_ids(ids); printf("]}\n");
    } else {
        double full = measure_bandwidth_default(d_src, d_sink, n_float4, totalSMs, iters, trials, buf_bytes, tma);
        printf("sm_count_requested,sm_count_allocated,bandwidth_GBps,pct_of_full_gpu,probe_observed_sm_count,tma_copy_only\n");
        printf("%u,%u,%.3f,100.0,%zu,%s\n", totalSMs, totalSMs, full, ids.size(),
               g_options.tma_copy_only ? "true" : "false");
        int minimum = cc_major >= 9 ? 8 : (cc_major == 8 ? 4 : (cc_major == 7 ? 2 : 1));
        int alignment = cc_major >= 9 ? 8 : (cc_major >= 7 ? 2 : 1);
        int step = g_options.sm_step ? std::max(g_options.sm_step, alignment) : alignment;
        std::vector<unsigned int> counts;
        for (unsigned int s = minimum; s < totalSMs; s += step) counts.push_back(s);
        std::mt19937 rng(g_options.seed);
        std::shuffle(counts.begin(), counts.end(), rng);
        for (unsigned int requested : counts) {
            unsigned int allocated = 0;
            double bw = measure_partition(requested, &allocated);
            printf("%u,%u,%.3f,%.1f,%u,%s\n", requested, allocated, bw, 100.0 * bw / full, g_probe_observed,
                   g_options.tma_copy_only ? "true" : "false");
            fflush(stdout);
        }
        double repeated = measure_bandwidth_default(d_src, d_sink, n_float4, totalSMs, iters, trials, buf_bytes, tma);
        fprintf(stderr, "Full-GPU baseline: start=%.3f end=%.3f GB/s drift=%.2f%%\n",
                full, repeated, 100.0 * (repeated / full - 1.0));
        fprintf(stderr, "95%% threshold is relative to this kernel/configuration; DRAM peak requires counter verification.\n");
    }
    if (tma.enabled) CUDA_RT_CHECK(cudaFree(tma.d_tensor_map));
    CUDA_RT_CHECK(cudaFree(d_src));
    CUDA_RT_CHECK(cudaFree(d_sink));
    return 0;
}
