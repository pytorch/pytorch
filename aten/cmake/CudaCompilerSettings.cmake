if(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
  add_definitions(-DTHC_EXPORTS)
endif()

list(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
list(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)

if(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
  include(select_compute_arch)
endif()

list(APPEND CUDA_NVCC_FLAGS $ENV{TORCH_NVCC_FLAGS})
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
if(CMAKE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
endif()

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
