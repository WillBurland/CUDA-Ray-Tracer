NVCC       = nvcc
FLAGS      = -O2 -std=c++17 -I./lib -MMD -MP -arch=sm_75 -diag-suppress 550

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

-include $(DEP)

.PHONY: all clean
