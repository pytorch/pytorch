cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(headeronly CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Main build file for torch/headeronly, except there's no build cuz this lib is header-only!

# ---[ Configure macro file.
set(C10_USE_GFLAGS ${USE_GFLAGS}) # used in cmake_macros.h.in
set(C10_USE_GLOG ${USE_GLOG}) # used in cmake_macros.h.in
set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
set(C10_USE_NUMA ${USE_NUMA}) # used in cmake_macros.h.in
set(C10_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME}) # used in cmake_macros.h.in
set(C10_USE_ROCM_KERNEL_ASSERT ${USE_ROCM_KERNEL_ASSERT}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/macros/cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/torch/headeronly/macros/cmake_macros.h)

file(GLOB HEADERONLY_HEADERS
    *.h
    cpu/**/*.h
    macros/*.h
    util/*.h
)

add_library(headeronly INTERFACE ${HEADERONLY_HEADERS})

install(FILES ${CMAKE_BINARY_DIR}/torch/headeronly/macros/cmake_macros.h
        DESTINATION include/torch/headeronly/macros)

if(NOT BUILD_LIBTORCHLESS)
  # ---[ Installation copied from c10/CMakeLists.txt
  install(TARGETS headeronly EXPORT Caffe2Targets DESTINATION lib)
endif()
