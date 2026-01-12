# CMakeLists.txt for _extern_triton bitcode generation
#
# This file configures the build of CUDA device functions to LLVM bitcode
# for use with Triton kernels via extern_libs mechanism.
#
# The bitcode is generated using clang++ and installed alongside PyTorch.

cmake_minimum_required(VERSION 3.18)

# Only build if CUDA is enabled and we have clang
if(NOT USE_CUDA)
  message(STATUS "Skipping extern_triton bitcode generation (CUDA not enabled)")
  return()
endif()

# Find clang++ for CUDA to LLVM bitcode compilation
find_program(CLANG_EXECUTABLE
  NAMES clang++ clang++-17 clang++-16 clang++-15 clang++-14
  HINTS
    ${LLVM_ROOT}/bin
    /usr/local/llvm/bin
    /usr/bin
  DOC "Path to clang++ for CUDA bitcode generation"
)

if(NOT CLANG_EXECUTABLE)
  message(STATUS "Skipping extern_triton bitcode generation (clang++ not found)")
  return()
endif()

message(STATUS "Found clang++ for bitcode generation: ${CLANG_EXECUTABLE}")

# Configuration
set(EXTERN_TRITON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EXTERN_TRITON_BC_OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Determine CUDA architecture - use the same as PyTorch build
# Default to sm_80 (Ampere) if not specified
if(TORCH_CUDA_ARCH_LIST)
  # Get the first architecture from the list
  list(GET TORCH_CUDA_ARCH_LIST 0 FIRST_ARCH)
  string(REGEX REPLACE "\\." "" CUDA_ARCH_NUM "${FIRST_ARCH}")
  set(EXTERN_TRITON_CUDA_ARCH "sm_${CUDA_ARCH_NUM}")
else()
  set(EXTERN_TRITON_CUDA_ARCH "sm_80")
endif()

message(STATUS "Building extern_triton bitcode for ${EXTERN_TRITON_CUDA_ARCH}")

# Clang flags for CUDA to LLVM bitcode compilation
# -fcuda-flush-denormals-to-zero is CRITICAL to avoid LLVM module flag conflicts
# with Triton's LLVMDialectModule during linking
set(EXTERN_TRITON_CLANG_FLAGS
  -x cuda
  --cuda-device-only
  -emit-llvm
  -c
  --cuda-gpu-arch=${EXTERN_TRITON_CUDA_ARCH}
  -O3
  -fcuda-flush-denormals-to-zero
  -I${CUDAToolkit_INCLUDE_DIRS}
)

# Source files to compile to bitcode
set(EXTERN_TRITON_CUDA_SOURCES
  elementwise_add.cu
)

# Output bitcode files
set(EXTERN_TRITON_BC_OUTPUTS "")

foreach(CUDA_SOURCE ${EXTERN_TRITON_CUDA_SOURCES})
  get_filename_component(SRC_NAME_WE ${CUDA_SOURCE} NAME_WE)
  set(BC_INPUT "${EXTERN_TRITON_SRC_DIR}/${CUDA_SOURCE}")
  set(BC_OUTPUT "${EXTERN_TRITON_BC_OUTPUT_DIR}/${SRC_NAME_WE}.bc")

  add_custom_command(
    OUTPUT ${BC_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${EXTERN_TRITON_BC_OUTPUT_DIR}
    COMMAND ${CLANG_EXECUTABLE} ${EXTERN_TRITON_CLANG_FLAGS} -o ${BC_OUTPUT} ${BC_INPUT}
    DEPENDS ${BC_INPUT}
    COMMENT "Generating LLVM bitcode: ${SRC_NAME_WE}.bc for ${EXTERN_TRITON_CUDA_ARCH}"
    VERBATIM
  )

  list(APPEND EXTERN_TRITON_BC_OUTPUTS ${BC_OUTPUT})
endforeach()

# Create a custom target for all bitcode files
add_custom_target(extern_triton_bitcode ALL
  DEPENDS ${EXTERN_TRITON_BC_OUTPUTS}
)

# Install the bitcode files to the torch library directory
install(
  FILES ${EXTERN_TRITON_BC_OUTPUTS}
  DESTINATION "${TORCH_INSTALL_LIB_DIR}"
  COMPONENT extern_triton
)

message(STATUS "extern_triton bitcode will be installed to: ${TORCH_INSTALL_LIB_DIR}")
