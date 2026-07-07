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
};

static constexpr size_t kTmaTileBytes = 4096;
static constexpr size_t kTmaTileFloat4s = kTmaTileBytes / sizeof(float4);

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
    __shared__ alignas(16) unsigned char smem_tile[kTmaTileBytes];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    float4 *tile = reinterpret_cast<float4 *>(smem_tile);

    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
    const size_t total_bytes = n_float4 * sizeof(float4);
    const size_t tile_bytes = kTmaTileBytes;
    const size_t tile_rows = 16;
    const size_t float4s_per_row = 16;

    for (size_t tile_row = blockIdx.x * tile_rows;
         tile_row * sizeof(float4) < total_bytes;
         tile_row += (size_t)gridDim.x * tile_rows) {
        if (threadIdx.x == 0) {
            auto token = cuda::device::barrier_arrive_tx(bar, 1, tile_bytes);
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                smem_tile, tensor_map, static_cast<int>(tile_row), 0, bar);
            bar.wait(std::move(token));
        }
        __syncthreads();

        for (size_t i = threadIdx.x; i < kTmaTileFloat4s; i += blockDim.x) {
            size_t global_float4 = tile_row * float4s_per_row + i;
            if (global_float4 < n_float4) {
                float4 v = tile[i];
                accum.x += v.x;
                accum.y += v.y;
                accum.z += v.z;
                accum.w += v.w;
            }
        }
        __syncthreads();
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
    const cuuint64_t global_dim[2] = { static_cast<cuuint64_t>(buf_bytes / 256), 256 };
    const cuuint64_t global_strides[1] = { 256 };
    const cuuint32_t box_dim[2] = { 16, 256 };
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
        read_bandwidth_kernel_tma<<<blocks, 256, 0, stream>>>(d_src, d_sink, n_float4, tma_ctx.d_tensor_map);
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
int main(int argc, char **argv)
{
    // ---- Parse arguments ----
    size_t buf_mb   = (argc > 1) ? atol(argv[1]) : 512;   // default 512 MB
    int    iters    = (argc > 2) ? atoi(argv[2]) : 20;
    int    dev_id   = (argc > 3) ? atoi(argv[3]) : 0;
    int    sm_step  = (argc > 4) ? atoi(argv[4]) : 0;     // 0 = auto
    int    trials   = (argc > 5) ? atoi(argv[5]) : 5;     // median of N trials
    int    mode     = (argc > 6) ? atoi(argv[6]) : 0;
    g_load_mode = static_cast<LoadMode>(std::max(0, std::min(mode, 4)));

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

    fprintf(stderr, "=== Green Context Bandwidth Saturation Benchmark ===\n");
    fprintf(stderr, "GPU:             %s\n", devName);
    fprintf(stderr, "Compute Cap:     %d.%d\n", cc_major, cc_minor);
    fprintf(stderr, "Total SMs:       %u\n", totalSMs);
    fprintf(stderr, "Buffer Size:     %zu MB\n", buf_mb);
    fprintf(stderr, "Iterations:      %d (per trial)\n", iters);
    fprintf(stderr, "Trials:          %d (median)\n", trials);
    fprintf(stderr, "Load Mode:       %s\n", load_mode_name(g_load_mode));

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
        if (!build_tma_context(cuDev, d_src, buf_bytes, &tma_ctx)) {
            fprintf(stderr, "Failed to build TMA tensor map\n");
            return EXIT_FAILURE;
        }
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
