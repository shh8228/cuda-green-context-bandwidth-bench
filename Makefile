# Green Context Bandwidth Saturation Benchmark
# Targets: NVIDIA Blackwell (GB200), sm_100
# Container: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

NVCC       ?= nvcc
CUDA_ARCH  ?= sm_100
CUDA_ROOT  ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
NVCC_FLAGS := -O3 -lineinfo -std=c++17 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr
# Development containers often expose libcuda.so.1 without the linker symlink.
# Stubs are for linking only; never add their directory to the runtime rpath.
LDFLAGS    := -L$(CUDA_ROOT)/lib64/stubs -L$(CUDA_ROOT)/lib/stubs -lcuda

TARGET     := green_ctx_bw_bench
SRC        := green_ctx_bw_bench.cu

.PHONY: all clean run run-large plot test test-gpu FORCE

all: $(TARGET)

# Flags/architectures may change without a source timestamp change.
FORCE:

$(TARGET): $(SRC) bench_options.h Makefile FORCE
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
	RUN_CUDA_TESTS=1 BENCHMARK_BINARY=$(abspath $(TARGET)) python3 -m unittest discover -s tests -v

clean:
	rm -f $(TARGET) results.csv *.png
