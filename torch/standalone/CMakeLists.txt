cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(c10 CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main build file for torch/standalone
# Note that torch/standalone is header only.

  # ---[ Configure macro file.
  set(C10_USE_GFLAGS ${USE_GFLAGS}) # used in cmake_macros.h.in
  set(C10_USE_GLOG ${USE_GLOG}) # used in cmake_macros.h.in
  set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
  set(C10_USE_NUMA ${USE_NUMA})
  set(C10_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
  set(C10_USE_ROCM_KERNEL_ASSERT ${USE_ROCM_KERNEL_ASSERT})
  configure_file(
      ${CMAKE_CURRENT_LIST_DIR}/macros/cmake_macros.h.in
      ${CMAKE_BINARY_DIR}/torch/standalone/macros/cmake_macros.h)

install(FILES ${CMAKE_BINARY_DIR}/torch/standalone/macros/cmake_macros.h
        DESTINATION include/torch/standalone/macros)
