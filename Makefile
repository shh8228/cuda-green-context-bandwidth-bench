# Green Context Bandwidth Saturation Benchmark
# Targets: NVIDIA Blackwell (GB200), sm_100
# Container: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

NVCC       ?= nvcc
CUDA_ARCH  ?= sm_100
NVCC_FLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) --expt-relaxed-constexpr
LDFLAGS    := -lcuda

TARGET     := green_ctx_bw_bench
SRC        := green_ctx_bw_bench.cu

.PHONY: all clean run plot

all: $(TARGET)

$(TARGET): $(SRC)
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

clean:
	rm -f $(TARGET) results.csv *.png
