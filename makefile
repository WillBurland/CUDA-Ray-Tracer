MAKEFLAGS += --no-print-directory

NVCC       = nvcc
FLAGS_BASE = -O2 -std=c++17 -I./lib -MMD -MP

FLAGS = $(FLAGS_BASE) \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_120,code=sm_120 \
	-gencode arch=compute_120,code=compute_120
	

SRC_DIR    = src
BUILD_DIR  = build
OBJ_DIR    = $(BUILD_DIR)/obj
TARGET     = raytracer
OUT        = $(BUILD_DIR)/$(TARGET)

SRC        = $(wildcard $(SRC_DIR)/*.cu)
OBJ        = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC))
DEP        = $(OBJ:.o=.d)

all: $(OUT)

$(OUT): $(OBJ) | $(BUILD_DIR)
	$(NVCC) $(FLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c $< -o $@

$(BUILD_DIR) $(OBJ_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d "." || echo 75)
dev:
	@$(MAKE) clean
	@$(MAKE) -j$(nproc) FLAGS="$(FLAGS_BASE) \
		-diag-suppress 550 \
		-Wno-deprecated-gpu-targets \
		-gencode arch=compute_$(ARCH),code=sm_$(ARCH)"
	@echo "Running build..."
	@$(MAKE) run

run: $(OUT)
	@./$(OUT)

help:
	@echo "Available targets:"
	@echo "  all    - Build release version (multi-arch)"
	@echo "  dev    - Cleans, builds (for your local GPU only), and runs"
	@echo "  run    - Run the compiled binary"
	@echo "  clean  - Remove build artifacts"

-include $(DEP)

.PHONY: all clean dev run help