cmake_minimum_required(VERSION 3.4.1)
project(pytorch CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(pytorch_android_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
set(libtorch_include_DIR ${pytorch_android_DIR}/libtorch_include/${ANDROID_ABI})

message(STATUS "libtorch dir:${libtorch_DIR}")

file(GLOB pytorch_android_SOURCES
  ${pytorch_android_DIR}/*.cpp
)

add_library(pytorch SHARED
    ${pytorch_android_SOURCES}
)

target_compile_options(pytorch PRIVATE
  -fexceptions
)

target_include_directories(pytorch PUBLIC
    ${libtorch_include_DIR}
)

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${BUILD_DIR}/fbjni/${ANDROID_ABI})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

function(import_static_lib name)
  add_library(${name} STATIC IMPORTED)
  set_property(
      TARGET ${name}
      PROPERTY IMPORTED_LOCATION
      ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/${name}.a)
endfunction(import_static_lib)

import_static_lib(libtorch)
import_static_lib(libc10)
import_static_lib(libnnpack)
import_static_lib(libqnnpack)
import_static_lib(libpytorch_qnnpack)
import_static_lib(libeigen_blas)
import_static_lib(libcpuinfo)
import_static_lib(libclog)

target_link_libraries(pytorch
    fbjni
    -Wl,--gc-sections
    -Wl,--whole-archive
    libtorch
    -Wl,--no-whole-archive
    libc10
    libnnpack
    libqnnpack
    libpytorch_qnnpack
    libeigen_blas
    libcpuinfo
    libclog
)
