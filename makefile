NVCC       = nvcc
FLAGS      = -O2 -std=c++17 -I./lib -MMD -MP -arch=sm_75

SRC_DIR    = src
BUILD_DIR  = build
OBJ_DIR    = $(BUILD_DIR)/obj
TARGET     = raytracer
OUT        = $(BUILD_DIR)/$(TARGET)

SRC        = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ        = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(filter %.cpp,$(SRC))) \
             $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(filter %.cu,$(SRC)))
DEP        = $(OBJ:.o=.d)

all: $(OUT)

$(OUT): $(OBJ) | $(BUILD_DIR)
	@$(NVCC) $(FLAGS) $^ -o $@ > /dev/null

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@$(NVCC) $(FLAGS) -c $< -o $@ > /dev/null

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	@$(NVCC) $(FLAGS) -c $< -o $@ > /dev/null

$(BUILD_DIR) $(OBJ_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)

-include $(DEP)

.PHONY: all clean
