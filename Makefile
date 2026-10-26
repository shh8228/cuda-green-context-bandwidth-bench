# Green Context Bandwidth Saturation Benchmark
# Targets: H200/Hopper (CUDA_ARCH=sm_90), B200/Blackwell (default sm_100)

NVCC       ?= nvcc
CUDA_ARCH  ?= sm_100
CUDA_ROOT  ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
NVCC_FLAGS := -O3 -lineinfo -std=c++17 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr
# Development containers often expose libcuda.so.1 without the linker symlink.
# Stubs are for linking only; never add their directory to the runtime rpath.
LDFLAGS    := -L$(CUDA_ROOT)/lib64/stubs -L$(CUDA_ROOT)/lib/stubs -lcuda

TARGET     := green_ctx_bw_bench
SRC        := green_ctx_bw_bench.cu

.PHONY: all clean run run-large plot test test-gpu test-ptx FORCE

all: $(TARGET)

# Flags/architectures may change without a source timestamp change.
FORCE:

$(TARGET): $(SRC) bench_options.h tma_tensor_core.cuh Makefile FORCE
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

# Default run: 512MB buffer, 20 iterations, GPU 0, auto SM step
run: $(TARGET)
	./$(TARGET) 512 20 0 0 | tee results.csv

# Run with larger buffer for more stable results
run-large: $(TARGET)
	./$(TARGET) 2048 30 0 0 | tee results.csv

# Generate plot from results
plot: results.csv
	python3 plot_results.py results.csv

test: $(TARGET)
	BENCHMARK_BINARY=$(abspath $(TARGET)) python3 -m unittest discover -s tests -v

test-gpu: $(TARGET)
	RUN_CUDA_TESTS=1 RUN_TCGEN05_TESTS=$(if $(filter sm_100a,$(CUDA_ARCH)),1,0) BENCHMARK_BINARY=$(abspath $(TARGET)) python3 -m unittest discover -s tests -v

# Verify descriptor-proxy acquire emission without a GPU.
test-ptx:
	@ptx=$$(mktemp); trap 'rm -f "$$ptx"' EXIT; \
	$(NVCC) $(NVCC_FLAGS) --ptx $(SRC) -o "$$ptx" && \
	BENCHMARK_PTX="$$ptx" python3 -m unittest discover -s tests -p test_tma_ptx.py -v

clean:
	rm -f $(TARGET) results.csv *.png
