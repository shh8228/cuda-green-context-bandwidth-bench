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
 *   ./green_ctx_bw_bench [buffer_size_MB] [iterations] [gpu_id] [sm_step] [trials]
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

// ---------------------------------------------------------------------------
// Measure bandwidth with a given stream, returning median over trials
// ---------------------------------------------------------------------------
static double measure_bandwidth_timed(cudaStream_t stream,
                                      const float4 *d_src, float *d_sink,
                                      size_t n_float4, int blocks, int iters,
                                      int trials, size_t buf_bytes)
{
    cudaEvent_t start, stop;
    CUDA_RT_CHECK(cudaEventCreate(&start));
    CUDA_RT_CHECK(cudaEventCreate(&stop));

    const int threads = 256;

    // Extended warmup (5 kernel launches)
    for (int i = 0; i < 5; ++i)
        read_bandwidth_kernel<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> bw_samples;
    bw_samples.reserve(trials);

    for (int t = 0; t < trials; ++t) {
        CUDA_RT_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iters; ++i)
            read_bandwidth_kernel<<<blocks, threads, 0, stream>>>(d_src, d_sink, n_float4);
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
                                        int iters, int trials, size_t buf_bytes)
{
    const int threads = 256;
    const int blocks = std::min((int)((n_float4 + threads - 1) / threads),
                                sm_count * 32);

    cudaStream_t stream;
    CUDA_RT_CHECK(cudaStreamCreate(&stream));

    double bw = measure_bandwidth_timed(stream, d_src, d_sink, n_float4,
                                        blocks, iters, trials, buf_bytes);

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
                                          int allocated_sms)
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
                                        blocks, iters, trials, buf_bytes);

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

    // ---- Baseline: full-GPU bandwidth ----
    double bw_full = measure_bandwidth_default(d_src, d_sink, n_float4,
                                                totalSMs, iters, trials,
                                                buf_bytes);
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
                                                 allocated);

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
    CUDA_RT_CHECK(cudaFree(d_src));
    CUDA_RT_CHECK(cudaFree(d_sink));

    return 0;
}
