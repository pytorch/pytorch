# Makefile for building torch_symm CUDA library to LLVM bitcode
#
# This Makefile compiles CUDA device functions to LLVM bitcode (.bc files)
# that can be linked with Triton kernels via extern_libs.
#
# Usage:
#   make                    # Build for default architecture (sm_80)
#   make CUDA_ARCH=sm_90    # Build for specific architecture
#   make test               # Build and verify bitcode creation
#   make clean              # Remove generated files

# Configuration
CUDA_ARCH ?= sm_80
CLANG ?= clang++
CUDA_PATH ?= /usr/local/cuda

# Source and output files
CUDA_SOURCES = torch_symm.cu
BC_OUTPUTS = $(CUDA_SOURCES:.cu=.bc)

# Clang flags for CUDA to LLVM bitcode compilation
# Note: -fcuda-flush-denormals-to-zero is required to avoid LLVM module flag
# conflicts with Triton's LLVMDialectModule during linking
CLANG_FLAGS = -x cuda \
              --cuda-device-only \
              -emit-llvm \
              -c \
              --cuda-gpu-arch=$(CUDA_ARCH) \
              -O3 \
              -fcuda-flush-denormals-to-zero \
              -I$(CUDA_PATH)/include

# Output directory (same as source by default)
BUILD_DIR = .

.PHONY: all clean test help

all: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

$(BUILD_DIR)/%.bc: %.cu
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(CLANG_FLAGS) $< -o $@
	@echo "Built $@ for $(CUDA_ARCH)"

# Individual targets
torch_symm.bc: $(BUILD_DIR)/torch_symm.bc

clean:
	rm -f $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

test: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))
	@echo "Bitcode files created successfully. Run Python tests separately."

# Architecture-specific targets
sm_70: CUDA_ARCH = sm_70
sm_70: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

sm_75: CUDA_ARCH = sm_75
sm_75: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

sm_80: CUDA_ARCH = sm_80
sm_80: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

sm_86: CUDA_ARCH = sm_86
sm_86: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

sm_89: CUDA_ARCH = sm_89
sm_89: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

sm_90: CUDA_ARCH = sm_90
sm_90: $(addprefix $(BUILD_DIR)/, $(BC_OUTPUTS))

# Help target
help:
	@echo "PyTorch External Triton Library Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all bitcode files for default architecture ($(CUDA_ARCH))"
	@echo "  torch_symm.bc        - Build torch symmetric memory bitcode"
	@echo "  clean                - Remove built files"
	@echo "  test                 - Build and verify bitcode creation"
	@echo "  sm_XX                - Build for specific CUDA architecture (70,75,80,86,89,90)"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH    - Target GPU architecture (default: sm_80)"
	@echo "  CLANG        - Clang compiler path (default: clang++)"
	@echo "  CUDA_PATH    - CUDA installation path (default: /usr/local/cuda)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all for sm_80 (Ampere)"
	@echo "  make CUDA_ARCH=sm_90    # Build all for sm_90 (Hopper)"
	@echo "  make torch_symm.bc      # Build only torch_symm"
	@echo "  make sm_70              # Build all for sm_70 (Volta)"
