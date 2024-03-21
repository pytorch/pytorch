# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include(GNUInstallDirs)

# ---[ Project and semantic versioning.
project(PYTORCH_QNNPACK C CXX ASM)

# ---[ Options.
option(PYTORCH_QNNPACK_CUSTOM_THREADPOOL "Build QNNPACK for custom thread pool" OFF)
set(PYTORCH_QNNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
set_property(CACHE PYTORCH_QNNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(PYTORCH_QNNPACK_BUILD_TESTS "Build QNNPACK unit tests" ON)
option(PYTORCH_QNNPACK_BUILD_BENCHMARKS "Build QNNPACK benchmarks" ON)

# Enable runtime requantization.
add_definitions(-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION=1)

# ---[ Target processor
SET(PYTORCH_QNNPACK_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64|arm64)$")
  SET(PYTORCH_QNNPACK_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
ENDIF()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # TODO: See https://github.com/pytorch/pytorch/issues/56285
  set_source_files_properties(src/operator-run.c PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(src/conv-run.cc PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(src/deconv-run.cc PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(src/fc-run.cc PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  set_source_files_properties(src/fc-dynamic-run.cc PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
endif()

# ---[ CMake options
if(PYTORCH_QNNPACK_BUILD_TESTS)
  enable_testing()
endif()

# ---[ Build flags
if(NOT CMAKE_SYSTEM_PROCESSOR)
  if(IOS)
    list(LENGTH IOS_ARCH IOS_ARCH_COUNT)
    if(IOS_ARCH_COUNT GREATER 1)
      message(FATAL_ERROR "Unsupported QNNPACK build with multiple iOS architectures (${IOS_ARCH}). "
        "Specify a single architecture in IOS_ARCH and re-configure. ")
    endif()
    if(NOT IOS_ARCH MATCHES "^(i386|x86_64|armv7.*|arm64.*)$")
      message(FATAL_ERROR "Unrecognized IOS_ARCH = ${IOS_ARCH}")
    endif()
  else()
    message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not defined")
  endif()
elseif(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64|armv[5-8].*|aarch64|arm64)$")
  message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_SYSTEM_NAME)
  message(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
elseif(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android)$")
  message(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
endif()

# ---[ Download deps
set(CONFU_DEPENDENCIES_SOURCE_DIR "${CMAKE_SOURCE_DIR}/deps"
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR "${CMAKE_BINARY_DIR}/deps"
  CACHE PATH "Confu-style dependencies binary directory")

if(NOT DEFINED CLOG_SOURCE_DIR)
  set(CLOG_SOURCE_DIR "${PROJECT_SOURCE_DIR}/deps/clog")
endif()

if(NOT USE_SYSTEM_CPUINFO)
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    message(STATUS "Downloading cpuinfo to ${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo (define CPUINFO_SOURCE_DIR to avoid it)")
    configure_file(cmake/DownloadCpuinfo.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
    set(CPUINFO_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo" CACHE STRING "cpuinfo source directory")
  endif()
endif()

if(NOT DEFINED FP16_SOURCE_DIR AND NOT USE_SYSTEM_FP16)
  message(STATUS "Downloading FP16 to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16 (define FP16_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadFP16.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  set(FP16_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16" CACHE STRING "FP16 source directory")
endif()

if(NOT DEFINED FXDIV_SOURCE_DIR AND NOT USE_SYSTEM_FXDIV)
  message(STATUS "Downloading FXdiv to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv (define FXDIV_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadFXdiv.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  set(FXDIV_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv" CACHE STRING "FXdiv source directory")
endif()

if(NOT DEFINED PSIMD_SOURCE_DIR AND NOT USE_SYSTEM_PSIMD)
  message(STATUS "Downloading PSimd to ${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd (define PSIMD_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadPSimd.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  set(PSIMD_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd" CACHE STRING "PSimd source directory")
endif()

if(NOT DEFINED PTHREADPOOL_SOURCE_DIR AND NOT USE_SYSTEM_PTHREADPOOL)
  message(STATUS "Downloading pthreadpool to ${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool (define PTHREADPOOL_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadPThreadPool.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  set(PTHREADPOOL_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool" CACHE STRING "pthreadpool source directory")
endif()

if(PYTORCH_QNNPACK_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  message(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  set(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
endif()

if(PYTORCH_QNNPACK_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
  message(STATUS "Downloading Google Benchmark to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
  configure_file(cmake/DownloadGoogleBenchmark.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  set(GOOGLEBENCHMARK_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")
endif()

# ---[ QNNPACK library
set(PYTORCH_QNNPACK_INIT_SRCS
  src/init.c
  src/add.c
  src/average-pooling.c
  src/channel-shuffle.c
  src/clamp.c
  src/conv-prepack.cc
  src/convolution.c
  src/deconvolution.c
  src/fc-prepack.cc
  src/fully-connected.c
  src/fully-connected-sparse.c
  src/global-average-pooling.c
  src/hardsigmoid.c
  src/hardswish.c
  src/leaky-relu.c
  src/max-pooling.c
  src/sigmoid.c
  src/softargmax.c
  src/tanh.c
  src/operator-delete.c)

set(PYTORCH_QNNPACK_EXEC_SRCS
  src/conv-run.cc
  src/deconv-run.cc
  src/fc-run.cc
  src/fc-unpack.cc
  src/fc-dynamic-run.cc
  src/indirection.c
  src/operator-run.c)

set(PYTORCH_QNNPACK_SCALAR_UKERNELS
  src/u8lut32norm/scalar.c
  src/x8lut/scalar.c)

set(PYTORCH_QNNPACK_PSIMD_UKERNELS
  src/sgemm/6x8-psimd.c)

set(PYTORCH_QNNPACK_ARM_NEON_UKERNELS
  src/q8avgpool/mp8x9p8q-neon.c
  src/q8avgpool/up8x9-neon.c
  src/q8avgpool/up8xm-neon.c
  src/q8conv/4x8-neon.c
  src/q8conv/8x8-neon.c
  src/q8dwconv/mp8x25-neon.c
  src/q8dwconv/mp8x25-neon-per-channel.c
  src/q8dwconv/mp8x27-neon.c
  src/q8dwconv/up8x9-neon.c
  src/q8dwconv/up8x9-neon-per-channel.c
  src/q8gavgpool/mp8x7p7q-neon.c
  src/q8gavgpool/up8x7-neon.c
  src/q8gavgpool/up8xm-neon.c
  src/q8gemm/4x-sumrows-neon.c
  src/q8gemm/4x8-neon.c
  src/q8gemm/4x8-dq-neon.c
  src/q8gemm/4x8c2-xzp-neon.c
  src/q8gemm/6x4-neon.c
  src/q8gemm/8x8-neon.c
  src/q8vadd/neon.c
  src/sgemm/5x8-neon.c
  src/sgemm/6x8-neon.c
  src/u8clamp/neon.c
  src/u8maxpool/16x9p8q-neon.c
  src/u8maxpool/sub16-neon.c
  src/u8rmax/neon.c
  src/x8zip/x2-neon.c
  src/x8zip/x3-neon.c
  src/x8zip/x4-neon.c
  src/x8zip/xm-neon.c)

set(PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS
  src/hgemm/8x8-aarch32-neonfp16arith.S
  src/q8conv/4x8-aarch32-neon.S
  src/q8dwconv/up8x9-aarch32-neon.S
  src/q8dwconv/up8x9-aarch32-neon-per-channel.S
  src/q8gemm/4x8-aarch32-neon.S
  src/q8gemm/4x8-dq-aarch32-neon.S
  src/q8gemm/4x8c2-xzp-aarch32-neon.S
  src/q8gemm_sparse/4x4-packA-aarch32-neon.S
  src/q8gemm_sparse/4x8c1x4-dq-packedA-aarch32-neon.S
  src/q8gemm_sparse/4x8c8x1-dq-packedA-aarch32-neon.S)

set(PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS
  src/q8conv/8x8-aarch64-neon.S
  src/q8gemm/8x8-aarch64-neon.S
  src/q8gemm/8x8-dq-aarch64-neon.S
  src/q8gemm_sparse/8x4-packA-aarch64-neon.S
  src/q8gemm_sparse/8x8c1x4-dq-packedA-aarch64-neon.S
  src/q8gemm_sparse/8x8c8x1-dq-packedA-aarch64-neon.S)

set(PYTORCH_QNNPACK_X86_SSE2_UKERNELS
  src/q8avgpool/mp8x9p8q-sse2.c
  src/q8avgpool/up8x9-sse2.c
  src/q8avgpool/up8xm-sse2.c
  src/q8conv/4x4c2-sse2.c
  src/q8dwconv/mp8x25-sse2.c
  src/q8dwconv/mp8x25-sse2-per-channel.c
  src/q8dwconv/mp8x27-sse2.c
  src/q8dwconv/up8x9-sse2.c
  src/q8dwconv/up8x9-sse2-per-channel.c
  src/q8gavgpool/mp8x7p7q-sse2.c
  src/q8gavgpool/up8x7-sse2.c
  src/q8gavgpool/up8xm-sse2.c
  src/q8gemm/2x4c8-sse2.c
  src/q8gemm/4x4c2-dq-sse2.c
  src/q8gemm/4x4c2-sse2.c
  src/q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c
  src/q8gemm_sparse/8x4-packA-sse2.c
  src/q8vadd/sse2.c
  src/u8clamp/sse2.c
  src/u8maxpool/16x9p8q-sse2.c
  src/u8maxpool/sub16-sse2.c
  src/u8rmax/sse2.c
  src/x8zip/x2-sse2.c
  src/x8zip/x3-sse2.c
  src/x8zip/x4-sse2.c
  src/x8zip/xm-sse2.c)

set(PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_SCALAR_UKERNELS} ${PYTORCH_QNNPACK_PSIMD_UKERNELS})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  list(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS})
  list(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS})
endif()
if(PYTORCH_QNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  list(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS})
  list(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS})
endif()
if(PYTORCH_QNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  list(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_X86_SSE2_UKERNELS})
endif()

if(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "default")
  add_library(pytorch_qnnpack ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
elseif(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "shared")
  add_library(pytorch_qnnpack SHARED ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
elseif(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "static")
  add_library(pytorch_qnnpack STATIC ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
else()
  message(FATAL_ERROR "Unsupported QNNPACK library type \"${PYTORCH_QNNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
endif()
set_target_properties(pytorch_qnnpack PROPERTIES
  CXX_STANDARD 14
  C_STANDARD 11
  C_EXTENSIONS YES)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  set_property(SOURCE ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm -mfpu=neon ")
  if(IOS)
    set_property(SOURCE ${PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  endif()
endif()
if(PYTORCH_QNNPACK_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)$" OR IOS_ARCH MATCHES "^arm64.*")
  set_property(SOURCE ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  if(IOS)
    set_property(SOURCE ${PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  endif()
endif()
if(PYTORCH_QNNPACK_TARGET_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  set_property(SOURCE ${PYTORCH_QNNPACK_X86_SSE2_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -msse2 ")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  set_property(SOURCE ${PYTORCH_QNNPACK_PSIMD_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm -mfpu=neon ")
  set_property(SOURCE ${PYTORCH_QNNPACK_SCALAR_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm ")
else()
  set_property(SOURCE ${PYTORCH_QNNPACK_PSIMD_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  set_property(SOURCE ${PYTORCH_QNNPACK_SCALAR_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
endif()
set_property(SOURCE ${PYTORCH_QNNPACK_INIT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Os ")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(SOURCE ${PYTORCH_QNNPACK_OPERATOR_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
endif()
target_include_directories(pytorch_qnnpack PUBLIC include)
target_include_directories(pytorch_qnnpack PUBLIC src)
set_target_properties(pytorch_qnnpack PROPERTIES PUBLIC_HEADER include/pytorch_qnnpack.h)
set_target_properties(pytorch_qnnpack PROPERTIES PUBLIC_HEADER include/qnnpack_func.h)

# ---[ Configure clog
if(NOT TARGET clog)
  set(CLOG_BUILD_TESTS OFF CACHE BOOL "")
  set(CLOG_RUNTIME_TYPE "${CPUINFO_RUNTIME_TYPE}" CACHE STRING "")
  add_subdirectory(
    "${CLOG_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/clog")
  # We build static version of clog but a dynamic library may indirectly depend on it
  set_property(TARGET clog PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(pytorch_qnnpack PUBLIC clog)

# ---[ Configure cpuinfo
if(NOT TARGET cpuinfo AND USE_SYSTEM_CPUINFO)
  add_library(cpuinfo SHARED IMPORTED)
  find_library(CPUINFO_LIBRARY cpuinfo)
  if(NOT CPUINFO_LIBRARY)
    message(FATAL_ERROR "Cannot find cpuinfo")
  endif()
  message("Found cpuinfo: ${CPUINFO_LIBRARY}")
  set_target_properties(cpuinfo PROPERTIES IMPORTED_LOCATION "${CPUINFO_LIBRARY}")
elseif(NOT TARGET cpuinfo)
  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${CPUINFO_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
endif()
target_link_libraries(pytorch_qnnpack PRIVATE cpuinfo)

# ---[ Configure pthreadpool
if(NOT TARGET pthreadpool AND NOT USE_SYSTEM_PTHREADPOOL)
  set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
  set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${PTHREADPOOL_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
elseif(NOT TARGET pthreadpool AND USE_SYSTEM_PTHREADPOOL)
  add_library(pthreadpool SHARED IMPORTED)
  find_library(PTHREADPOOL_LIBRARY pthreadpool)
  if(NOT PTHREADPOOL_LIBRARY)
    message(FATAL_ERROR "Cannot find pthreadpool")
  endif()
  message("-- Found pthreadpool: ${PTHREADPOOL_LIBRARY}")
  set_target_properties(pthreadpool PROPERTIES
    IMPORTED_LOCATION "${PTHREADPOOL_LIBRARY}")
  add_library(pthreadpool_interface INTERFACE)
endif()
if(PYTORCH_QNNPACK_CUSTOM_THREADPOOL)
  # Depend on pthreadpool interface, but not on implementation.
  # This is used when QNNPACK user (e.g. Caffe2) provides its own threadpool implementation.
  target_link_libraries(pytorch_qnnpack PUBLIC pthreadpool_interface)
else()
  target_link_libraries(pytorch_qnnpack PUBLIC pthreadpool)
endif()

# ---[ Configure FXdiv
if(NOT TARGET fxdiv AND NOT USE_SYSTEM_FXDIV)
  set(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
  set(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${FXDIV_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv")
elseif(NOT TARGET fxdiv AND USE_SYSTEM_FXDIV)
  find_file(FXDIV_HDR fxdiv.h PATH_SUFFIXES include)
  if(NOT FXDIV_HDR)
    message(FATAL_ERROR "Cannot find fxdiv")
  endif()
  add_library(fxdiv STATIC "${FXDIV_HDR}")
  set_property(TARGET fxdiv PROPERTY LINKER_LANGUAGE C)
endif()
target_link_libraries(pytorch_qnnpack PRIVATE fxdiv)

# ---[ Configure psimd
if(NOT TARGET psimd AND NOT USE_SYSTEM_PSIMD)
  add_subdirectory(
    "${PSIMD_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd")
elseif(NOT TARGET psimd AND USE_SYSTEM_PSIMD)
  find_file(PSIMD_HDR psimd.h PATH_SUFFIXES include)
  if(NOT PSIMD_HDR)
    message(FATAL_ERROR "Cannot find psimd")
  endif()
  add_library(psimd STATIC "${PSIMD_HDR}")
  set_property(TARGET psimd PROPERTY LINKER_LANGUAGE C)
endif()
target_link_libraries(pytorch_qnnpack PRIVATE psimd)

# ---[ Configure FP16
if(NOT TARGET fp16 AND NOT USE_SYSTEM_FP16)
  set(FP16_BUILD_TESTS OFF CACHE BOOL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${FP16_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16")
elseif(NOT TARGET fp16 AND USE_SYSTEM_FP16)
  find_file(FP16_HDR fp16.h PATH_SUFFIXES include)
  if(NOT FP16_HDR)
    message(FATAL_ERROR "Cannot find fp16")
  endif()
  add_library(fp16 STATIC "${FP16_HDR}")
  set_target_properties(fp16 PROPERTIES LINKER_LANGUAGE C)
endif()
target_link_libraries(pytorch_qnnpack PRIVATE fp16)

install(TARGETS pytorch_qnnpack
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---[ QNNPACK unit tests
if(PYTORCH_QNNPACK_BUILD_TESTS)
  # ---[ Build google test
  if(NOT TARGET gtest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  endif()

  # ---[ Build unit tests for high-level functionality
  add_executable(convolution-test test/convolution.cc)
  set_target_properties(convolution-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(convolution-test PRIVATE src test)
  target_link_libraries(convolution-test PRIVATE pytorch_qnnpack clog cpuinfo fp16 gtest gtest_main)
  add_test(convolution-test convolution-test)

  add_executable(deconvolution-test test/deconvolution.cc)
  set_target_properties(deconvolution-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(deconvolution-test PRIVATE src test)
  target_link_libraries(deconvolution-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(deconvolution-test deconvolution-test)

  add_executable(fully-connected-test test/fully-connected.cc)
  set_target_properties(fully-connected-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(fully-connected-test PRIVATE src test)
  target_link_libraries(fully-connected-test PRIVATE pytorch_qnnpack clog cpuinfo fp16 gtest gtest_main)
  add_test(fully-connected-test fully-connected-test)

  add_executable(fully-connected-sparse-test test/fully-connected-sparse.cc)
  set_target_properties(fully-connected-sparse-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(fully-connected-sparse-test PRIVATE src test)
  target_link_libraries(fully-connected-sparse-test PRIVATE pytorch_qnnpack clog cpuinfo fp16 gtest gtest_main)
  add_test(fully-connected-sparse-test fully-connected-sparse-test)

  add_executable(channel-shuffle-test test/channel-shuffle.cc)
  set_target_properties(channel-shuffle-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(channel-shuffle-test PRIVATE src test)
  target_link_libraries(channel-shuffle-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(channel-shuffle-test channel-shuffle-test)

  add_executable(add-test test/add.cc)
  set_target_properties(add-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(add-test PRIVATE src test)
  target_link_libraries(add-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(add-test add-test)

  add_executable(leaky-relu-test test/leaky-relu.cc)
  set_target_properties(leaky-relu-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(leaky-relu-test PRIVATE src test)
  target_link_libraries(leaky-relu-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(leaky-relu-test leaky-relu-test)

  add_executable(sigmoid-test test/sigmoid.cc)
  set_target_properties(sigmoid-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(sigmoid-test PRIVATE src test)
  target_link_libraries(sigmoid-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(sigmoid-test sigmoid-test)

  add_executable(clamp-test test/clamp.cc)
  set_target_properties(clamp-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(clamp-test PRIVATE src test)
  target_link_libraries(clamp-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(clamp-test clamp-test)

  add_executable(softargmax-test test/softargmax.cc)
  set_target_properties(softargmax-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(softargmax-test PRIVATE src test)
  target_link_libraries(softargmax-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(softargmax-test softargmax-test)

  add_executable(tanh-test test/tanh.cc)
  set_target_properties(tanh-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(tanh-test PRIVATE src test)
  target_link_libraries(tanh-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(tanh-test tanh-test)

  add_executable(hardsigmoid-test test/hardsigmoid.cc)
  set_target_properties(hardsigmoid-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(hardsigmoid-test PRIVATE src test)
  target_link_libraries(hardsigmoid-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(hardsigmoid-test hardsigmoid-test)

  add_executable(hardswish-test test/hardswish.cc)
  set_target_properties(hardswish-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(hardswish-test PRIVATE src test)
  target_link_libraries(hardswish-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(hardswish-test hardswish-test)

  add_executable(max-pooling-test test/max-pooling.cc)
  set_target_properties(max-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(max-pooling-test PRIVATE src test)
  target_link_libraries(max-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(max-pooling-test max-pooling-test)

  add_executable(average-pooling-test test/average-pooling.cc)
  set_target_properties(average-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(average-pooling-test PRIVATE src test)
  target_link_libraries(average-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(average-pooling-test average-pooling-test)

  add_executable(global-average-pooling-test test/global-average-pooling.cc)
  set_target_properties(global-average-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(global-average-pooling-test PRIVATE src test)
  target_link_libraries(global-average-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  add_test(global-average-pooling-test global-average-pooling-test)

  # ---[ Build unit tests for micro-kernels
  add_executable(q8gemm-test test/q8gemm.cc)
  set_target_properties(q8gemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8gemm-test PRIVATE src test)
  target_link_libraries(q8gemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8gemm-test q8gemm-test)

  add_executable(q8gemm-sparse-test test/q8gemm_sparse.cc)
  set_target_properties(q8gemm-sparse-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8gemm-sparse-test PRIVATE src test)
  target_link_libraries(q8gemm-sparse-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8gemm-sparse-test q8gemm-sparse-test)

  add_executable(q8conv-test test/q8conv.cc)
  set_target_properties(q8conv-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8conv-test PRIVATE src test)
  target_link_libraries(q8conv-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8conv-test q8conv-test)

  add_executable(q8dwconv-test test/q8dwconv.cc)
  set_target_properties(q8dwconv-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8dwconv-test PRIVATE src test)
  target_link_libraries(q8dwconv-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8dwconv-test q8dwconv-test)

  add_executable(q8vadd-test test/q8vadd.cc)
  set_target_properties(q8vadd-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8vadd-test PRIVATE src test)
  target_link_libraries(q8vadd-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8vadd-test q8vadd-test)

  add_executable(q8avgpool-test test/q8avgpool.cc)
  set_target_properties(q8avgpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8avgpool-test PRIVATE src test)
  target_link_libraries(q8avgpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8avgpool-test q8avgpool-test)

  add_executable(q8gavgpool-test test/q8gavgpool.cc)
  set_target_properties(q8gavgpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8gavgpool-test PRIVATE src test)
  target_link_libraries(q8gavgpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(q8gavgpool-test q8gavgpool-test)

  add_executable(u8maxpool-test test/u8maxpool.cc)
  set_target_properties(u8maxpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(u8maxpool-test PRIVATE src test)
  target_link_libraries(u8maxpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(u8maxpool-test u8maxpool-test)

  add_executable(u8clamp-test test/u8clamp.cc)
  set_target_properties(u8clamp-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(u8clamp-test PRIVATE src test)
  target_link_libraries(u8clamp-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(u8clamp-test u8clamp-test)

  add_executable(u8rmax-test test/u8rmax.cc)
  set_target_properties(u8rmax-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(u8rmax-test PRIVATE src test)
  target_link_libraries(u8rmax-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(u8rmax-test u8rmax-test)

  add_executable(u8lut32norm-test test/u8lut32norm.cc)
  set_target_properties(u8lut32norm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(u8lut32norm-test PRIVATE src test)
  target_link_libraries(u8lut32norm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(u8lut32norm-test u8lut32norm-test)

  add_executable(x8lut-test test/x8lut.cc)
  set_target_properties(x8lut-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(x8lut-test PRIVATE src test)
  target_link_libraries(x8lut-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(x8lut-test x8lut-test)

  add_executable(x8zip-test test/x8zip.cc)
  set_target_properties(x8zip-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(x8zip-test PRIVATE src test)
  target_link_libraries(x8zip-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(x8zip-test x8zip-test)

  add_executable(hgemm-test test/hgemm.cc)
  set_target_properties(hgemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(hgemm-test PRIVATE src test)
  target_link_libraries(hgemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(hgemm-test hgemm-test)

  add_executable(sgemm-test test/sgemm.cc)
  set_target_properties(sgemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(sgemm-test PRIVATE src test)
  target_link_libraries(sgemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  add_test(sgemm-test sgemm-test)
endif()

# ---[ QNNPACK micro-benchmarks
if(PYTORCH_QNNPACK_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  if(NOT TARGET benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    add_subdirectory(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")
  endif()

  add_executable(add-bench bench/add.cc)
  set_target_properties(add-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(add-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(average-pooling-bench bench/average-pooling.cc)
  set_target_properties(average-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(average-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(channel-shuffle-bench bench/channel-shuffle.cc)
  set_target_properties(channel-shuffle-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(channel-shuffle-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(convolution-bench bench/convolution.cc)
  set_target_properties(convolution-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(convolution-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(global-average-pooling-bench bench/global-average-pooling.cc)
  set_target_properties(global-average-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(global-average-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(max-pooling-bench bench/max-pooling.cc)
  set_target_properties(max-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(max-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(sigmoid-bench bench/sigmoid.cc)
  set_target_properties(sigmoid-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(sigmoid-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(softargmax-bench bench/softargmax.cc)
  set_target_properties(softargmax-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(softargmax-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(tanh-bench bench/tanh.cc)
  set_target_properties(tanh-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(tanh-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(hardsigmoid-bench bench/hardsigmoid.cc)
  set_target_properties(hardsigmoid-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(hardsigmoid-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(hardswish-bench bench/hardswish.cc)
  set_target_properties(hardswish-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_link_libraries(hardswish-bench PRIVATE pytorch_qnnpack benchmark)

  add_executable(q8gemm-bench bench/q8gemm.cc)
  set_target_properties(q8gemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8gemm-bench PRIVATE src)
  target_compile_definitions(q8gemm-bench PRIVATE pytorch_PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP=0)
  target_link_libraries(q8gemm-bench PRIVATE pytorch_qnnpack cpuinfo fp16 benchmark)

  add_executable(q8gemm-sparse-bench bench/q8gemm_sparse.cc)
  set_target_properties(q8gemm-sparse-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(q8gemm-sparse-bench PRIVATE src)
  target_link_libraries(q8gemm-sparse-bench PRIVATE pytorch_qnnpack cpuinfo fp16 benchmark)

  add_executable(hgemm-bench bench/hgemm.cc)
  set_target_properties(hgemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(hgemm-bench PRIVATE src)
  target_link_libraries(hgemm-bench PRIVATE pytorch_qnnpack cpuinfo fp16 benchmark)

  add_executable(sgemm-bench bench/sgemm.cc)
  set_target_properties(sgemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories(sgemm-bench PRIVATE src)
  target_link_libraries(sgemm-bench PRIVATE pytorch_qnnpack cpuinfo benchmark)
endif()
