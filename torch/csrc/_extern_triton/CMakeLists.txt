# CMakeLists.txt for _extern_triton bitcode generation and NCCL kernel verification
#
# This file configures:
# 1. Build of CUDA device functions to LLVM bitcode for use with Triton kernels
#    via extern_libs mechanism.
# 2. Compilation verification of NCCL device kernels (nccl_symm.cuh) using nvcc
#    to ensure the NCCL backend code compiles correctly.
#
# The bitcode is generated using clang++ and installed alongside PyTorch.
#
# ARCHITECTURE:
# - symm_comm.cuh: Common context/team structures and NVSHMEM device function declarations
# - nccl_symm.cuh: NCCL backend implementations (compiled via nvcc for verification)
# - nvshmem_symm.cuh: NVSHMEM backend implementations (always compiled)
# - torch_symm.cu: Unified frontend that dispatches to backends
#
# BUILD MODES:
# - Bitcode build (clang++): TORCH_SYMM_BITCODE_BUILD=1 is defined, NCCL device
#   code is excluded to avoid unresolved symbols when linking with Triton.
# - CUDA verification build (nvcc): NCCL device code is included and compiled
#   to verify it builds correctly with NCCL headers.
#
# The NVSHMEM device functions declared in symm_comm.cuh are resolved at Triton
# compile time by linking with libnvshmem_device.bc (from NVSHMEM installation).
# Our torch_symm.bc provides the wrapper functions that Triton kernels call via
# extern_elementwise.

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
# -DTORCH_SYMM_BITCODE_BUILD=1 excludes NCCL device code (which would cause
# unresolved symbols when linked with Triton kernels)
set(EXTERN_TRITON_CLANG_FLAGS
  -x cuda
  --cuda-device-only
  -emit-llvm
  -c
  --cuda-gpu-arch=${EXTERN_TRITON_CUDA_ARCH}
  -O3
  -fcuda-flush-denormals-to-zero
  -DTORCH_SYMM_BITCODE_BUILD=1
  -I${CUDAToolkit_INCLUDE_DIRS}
)

# Header files (for dependency tracking)
set(EXTERN_TRITON_HEADERS
  ${EXTERN_TRITON_SRC_DIR}/symm_comm.cuh
  ${EXTERN_TRITON_SRC_DIR}/nccl_symm.cuh
  ${EXTERN_TRITON_SRC_DIR}/nvshmem_symm.cuh
)

# Source files to compile to bitcode
set(EXTERN_TRITON_CUDA_SOURCES
  torch_symm.cu
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
    DEPENDS ${BC_INPUT} ${EXTERN_TRITON_HEADERS}
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

# =============================================================================
# NCCL DEVICE CODE VERIFICATION BUILD
# =============================================================================
#
# This section compiles nccl_symm.cuh using nvcc to verify that the NCCL device
# code compiles correctly during the PyTorch build process.
#
# Note: This is separate from the bitcode build (which uses clang++ and defines
# TORCH_SYMM_BITCODE_BUILD=1 to exclude NCCL code). Here we compile with nvcc
# to ensure the NCCL backend implementation is valid.

# Check if NCCL is available for verification build
if(USE_NCCL AND NCCL_FOUND)
  message(STATUS "Building NCCL symm verification target")

  # Create a simple .cu file that includes nccl_symm.cuh to verify it compiles
  set(NCCL_VERIFY_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/nccl_symm_verify.cu")
  file(WRITE ${NCCL_VERIFY_SOURCE} "
// Auto-generated verification file for nccl_symm.cuh compilation
// This file is used to verify that NCCL device code compiles correctly with nvcc

#include \"${EXTERN_TRITON_SRC_DIR}/nccl_symm.cuh\"

// Dummy kernel to ensure device code is instantiated
__global__ void nccl_symm_verify_kernel() {
  // This kernel exists only to verify compilation
  // The actual NCCL device functions are in nccl_symm.cuh
}
")

  # Create object library for NCCL verification
  add_library(nccl_symm_verify OBJECT ${NCCL_VERIFY_SOURCE})

  # Set CUDA properties
  set_target_properties(nccl_symm_verify PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # Include directories
  target_include_directories(nccl_symm_verify PRIVATE
    ${EXTERN_TRITON_SRC_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/torch/csrc/distributed/c10d/symm_mem
  )

  # Add NCCL device headers if available
  # The NCCL_SYMM_TYPES_AVAILABLE macro is defined when NCCL provides device types
  # For now, we compile without it to verify the fallback path
  target_compile_definitions(nccl_symm_verify PRIVATE
    # Don't define NCCL_SYMM_TYPES_AVAILABLE - this verifies the fallback path compiles
    # When NCCL device headers are available, define it to verify full compilation
  )

  # Make the bitcode target depend on NCCL verification
  add_dependencies(extern_triton_bitcode nccl_symm_verify)

  message(STATUS "NCCL symm verification target configured")
else()
  message(STATUS "Skipping NCCL symm verification (NCCL not available)")
endif()
