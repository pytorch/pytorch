if(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
  add_definitions(-DTHC_EXPORTS)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9")
    if(CUDA_VERSION VERSION_LESS "8.0")
      message(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

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
