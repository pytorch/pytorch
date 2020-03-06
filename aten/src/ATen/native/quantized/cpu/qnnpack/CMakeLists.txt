# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(PYTORCH_QNNPACK C CXX ASM)

# ---[ Options.
OPTION(PYTORCH_QNNPACK_CUSTOM_THREADPOOL "Build QNNPACK for custom thread pool" OFF)
SET(PYTORCH_QNNPACK_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE PYTORCH_QNNPACK_LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(PYTORCH_QNNPACK_BUILD_TESTS "Build QNNPACK unit tests" ON)
OPTION(PYTORCH_QNNPACK_BUILD_BENCHMARKS "Build QNNPACK benchmarks" ON)

# Enable runtime requantization.
ADD_DEFINITIONS(-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION=1)

# ---[ CMake options
IF(PYTORCH_QNNPACK_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Build flags
IF(NOT CMAKE_SYSTEM_PROCESSOR)
  IF(IOS)
    LIST(LENGTH IOS_ARCH IOS_ARCH_COUNT)
    IF(IOS_ARCH_COUNT GREATER 1)
      MESSAGE(FATAL_ERROR "Unsupported QNNPACK build with multiple iOS architectures (${IOS_ARCH}). "
        "Specify a single architecture in IOS_ARCH and re-configure. ")
    ENDIF()
    IF(NOT IOS_ARCH MATCHES "^(i386|x86_64|armv7.*|arm64.*)$")
      MESSAGE(FATAL_ERROR "Unrecognized IOS_ARCH = ${IOS_ARCH}")
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not defined")
  ENDIF()
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64|armv[5-8].*|aarch64)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux|Android)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
ENDIF()

# ---[ Download deps
SET(CONFU_DEPENDENCIES_SOURCE_DIR "${CMAKE_SOURCE_DIR}/deps"
  CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR "${CMAKE_BINARY_DIR}/deps"
  CACHE PATH "Confu-style dependencies binary directory")

IF(NOT DEFINED CLOG_SOURCE_DIR)
  SET(CLOG_SOURCE_DIR "${PROJECT_SOURCE_DIR}/deps/clog")
ENDIF()

IF(NOT DEFINED CPUINFO_SOURCE_DIR)
  MESSAGE(STATUS "Downloading cpuinfo to ${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo (define CPUINFO_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadCpuinfo.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo-download")
  SET(CPUINFO_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/cpuinfo" CACHE STRING "cpuinfo source directory")
ENDIF()

IF(NOT DEFINED FP16_SOURCE_DIR)
  MESSAGE(STATUS "Downloading FP16 to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16 (define FP16_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadFP16.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16-download")
  SET(FP16_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fp16" CACHE STRING "FP16 source directory")
ENDIF()

IF(NOT DEFINED FXDIV_SOURCE_DIR)
  MESSAGE(STATUS "Downloading FXdiv to ${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv (define FXDIV_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadFXdiv.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv-download")
  SET(FXDIV_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/fxdiv" CACHE STRING "FXdiv source directory")
ENDIF()

IF(NOT DEFINED PSIMD_SOURCE_DIR)
  MESSAGE(STATUS "Downloading PSimd to ${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd (define PSIMD_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPSimd.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd-download")
  SET(PSIMD_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/psimd" CACHE STRING "PSimd source directory")
ENDIF()

IF(NOT DEFINED PTHREADPOOL_SOURCE_DIR)
  MESSAGE(STATUS "Downloading pthreadpool to ${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool (define PTHREADPOOL_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadPThreadPool.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool-download")
  SET(PTHREADPOOL_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/pthreadpool" CACHE STRING "pthreadpool source directory")
ENDIF()

IF(PYTORCH_QNNPACK_BUILD_TESTS AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
  SET(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
ENDIF()

IF(PYTORCH_QNNPACK_BUILD_BENCHMARKS AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
  MESSAGE(STATUS "Downloading Google Benchmark to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
  CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
  SET(GOOGLEBENCHMARK_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")
ENDIF()

# ---[ QNNPACK library
SET(PYTORCH_QNNPACK_INIT_SRCS
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
  src/global-average-pooling.c
  src/leaky-relu.c
  src/max-pooling.c
  src/sigmoid.c
  src/softargmax.c
  src/tanh.c
  src/operator-delete.c)

SET(PYTORCH_QNNPACK_EXEC_SRCS
  src/conv-run.cc
  src/fc-run.cc
  src/fc-dynamic-run.cc
  src/indirection.c
  src/operator-run.c)

SET(PYTORCH_QNNPACK_SCALAR_UKERNELS
  src/u8lut32norm/scalar.c
  src/x8lut/scalar.c)

SET(PYTORCH_QNNPACK_PSIMD_UKERNELS
  src/sgemm/6x8-psimd.c)

SET(PYTORCH_QNNPACK_ARM_NEON_UKERNELS
  src/q8avgpool/mp8x9p8q-neon.c
  src/q8avgpool/up8x9-neon.c
  src/q8avgpool/up8xm-neon.c
  src/q8conv/4x8-neon.c
  src/q8conv/8x8-neon.c
  src/q8dwconv/mp8x25-neon.c
  src/q8dwconv/up8x9-neon.c
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

SET(PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS
  src/hgemm/8x8-aarch32-neonfp16arith.S
  src/q8conv/4x8-aarch32-neon.S
  src/q8dwconv/up8x9-aarch32-neon.S
  src/q8gemm/4x8-aarch32-neon.S
  src/q8gemm/4x8-dq-aarch32-neon.S
  src/q8gemm/4x8c2-xzp-aarch32-neon.S)

SET(PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS
  src/q8conv/8x8-aarch64-neon.S
  src/q8gemm/8x8-aarch64-neon.S
  src/q8gemm/8x8-dq-aarch64-neon.S)

SET(PYTORCH_QNNPACK_X86_SSE2_UKERNELS
  src/q8avgpool/mp8x9p8q-sse2.c
  src/q8avgpool/up8x9-sse2.c
  src/q8avgpool/up8xm-sse2.c
  src/q8conv/4x4c2-sse2.c
  src/q8dwconv/mp8x25-sse2.c
  src/q8dwconv/up8x9-sse2.c
  src/q8gavgpool/mp8x7p7q-sse2.c
  src/q8gavgpool/up8x7-sse2.c
  src/q8gavgpool/up8xm-sse2.c
  src/q8gemm/2x4c8-sse2.c
  src/q8gemm/4x4c2-dq-sse2.c
  src/q8gemm/4x4c2-sse2.c
  src/q8vadd/sse2.c
  src/u8clamp/sse2.c
  src/u8maxpool/16x9p8q-sse2.c
  src/u8maxpool/sub16-sse2.c
  src/u8rmax/sse2.c
  src/x8zip/x2-sse2.c
  src/x8zip/x3-sse2.c
  src/x8zip/x4-sse2.c
  src/x8zip/xm-sse2.c)

SET(PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_SCALAR_UKERNELS} ${PYTORCH_QNNPACK_PSIMD_UKERNELS})
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  LIST(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS})
  LIST(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  LIST(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS})
  LIST(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS})
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  LIST(APPEND PYTORCH_QNNPACK_UKERNELS ${PYTORCH_QNNPACK_X86_SSE2_UKERNELS})
ENDIF()

IF(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "default")
  ADD_LIBRARY(pytorch_qnnpack ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
ELSEIF(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "shared")
  ADD_LIBRARY(pytorch_qnnpack SHARED ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
ELSEIF(PYTORCH_QNNPACK_LIBRARY_TYPE STREQUAL "static")
  ADD_LIBRARY(pytorch_qnnpack STATIC ${PYTORCH_QNNPACK_INIT_SRCS} ${PYTORCH_QNNPACK_EXEC_SRCS} ${PYTORCH_QNNPACK_UKERNELS})
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported QNNPACK library type \"${PYTORCH_QNNPACK_LIBRARY_TYPE}\". Must be \"static\", \"shared\", or \"default\"")
ENDIF()
SET_TARGET_PROPERTIES(pytorch_qnnpack PROPERTIES
  CXX_STANDARD 14
  C_STANDARD 11
  C_EXTENSIONS YES)
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm -mfpu=neon ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_AARCH32_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR IOS_ARCH MATCHES "^arm64.*")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_ARM_NEON_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  IF(IOS)
    SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_AARCH64_ASM_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -arch ${IOS_ARCH} ")
  ENDIF()
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86_64)$" OR IOS_ARCH MATCHES "^(i386|x86_64)$")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_X86_SSE2_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -msse2 ")
ENDIF()
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]" OR IOS_ARCH MATCHES "^armv7")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_PSIMD_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm -mfpu=neon ")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_SCALAR_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 -marm ")
ELSE()
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_PSIMD_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_SCALAR_UKERNELS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
ENDIF()
SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_INIT_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -Os ")
IF(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  SET_PROPERTY(SOURCE ${PYTORCH_QNNPACK_OPERATOR_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -O2 ")
ENDIF()
TARGET_INCLUDE_DIRECTORIES(pytorch_qnnpack PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(pytorch_qnnpack PUBLIC src)
SET_TARGET_PROPERTIES(pytorch_qnnpack PROPERTIES PUBLIC_HEADER include/pytorch_qnnpack.h)
SET_TARGET_PROPERTIES(pytorch_qnnpack PROPERTIES PUBLIC_HEADER include/conv_utils.h)
SET_TARGET_PROPERTIES(pytorch_qnnpack PROPERTIES PUBLIC_HEADER include/qnnpack_func.h)

# ---[ Configure clog
IF(NOT TARGET clog)
  SET(CLOG_BUILD_TESTS OFF CACHE BOOL "")
  SET(CLOG_RUNTIME_TYPE "${CPUINFO_RUNTIME_TYPE}" CACHE STRING "")
  ADD_SUBDIRECTORY(
    "${CLOG_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/clog")
  # We build static version of clog but a dynamic library may indirectly depend on it
  SET_PROPERTY(TARGET clog PROPERTY POSITION_INDEPENDENT_CODE ON)
ENDIF()
TARGET_LINK_LIBRARIES(pytorch_qnnpack PUBLIC clog)

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${CPUINFO_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
ENDIF()
TARGET_LINK_LIBRARIES(pytorch_qnnpack PRIVATE cpuinfo)

# ---[ Configure pthreadpool
IF(NOT TARGET pthreadpool)
  SET(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "")
  SET(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${PTHREADPOOL_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/pthreadpool")
ENDIF()
IF(PYTORCH_QNNPACK_CUSTOM_THREADPOOL)
  # Depend on pthreadpool interface, but not on implementation.
  # This is used when QNNPACK user (e.g. Caffe2) provides its own threadpool implementation.
  TARGET_LINK_LIBRARIES(pytorch_qnnpack PUBLIC pthreadpool_interface)
ELSE()
  TARGET_LINK_LIBRARIES(pytorch_qnnpack PUBLIC pthreadpool)
ENDIF()

# ---[ Configure FXdiv
IF(NOT TARGET fxdiv)
  SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
  SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FXDIV_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fxdiv")
ENDIF()
TARGET_LINK_LIBRARIES(pytorch_qnnpack PRIVATE fxdiv)

# ---[ Configure psimd
IF(NOT TARGET psimd)
  ADD_SUBDIRECTORY(
    "${PSIMD_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/psimd")
ENDIF()
TARGET_LINK_LIBRARIES(pytorch_qnnpack PRIVATE psimd)

# ---[ Configure FP16
IF(NOT TARGET fp16)
  SET(FP16_BUILD_TESTS OFF CACHE BOOL "")
  SET(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY(
    "${FP16_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/fp16")
ENDIF()
TARGET_LINK_LIBRARIES(pytorch_qnnpack PRIVATE fp16)

INSTALL(TARGETS pytorch_qnnpack
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---[ QNNPACK unit tests
IF(PYTORCH_QNNPACK_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  ENDIF()

  # ---[ Build unit tests for high-level functionality
  ADD_EXECUTABLE(convolution-test test/convolution.cc)
  SET_TARGET_PROPERTIES(convolution-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(convolution-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(convolution-test PRIVATE pytorch_qnnpack clog cpuinfo fp16 gtest gtest_main)
  ADD_TEST(convolution-test convolution-test)

  ADD_EXECUTABLE(deconvolution-test test/deconvolution.cc)
  SET_TARGET_PROPERTIES(deconvolution-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(deconvolution-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(deconvolution-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(deconvolution-test deconvolution-test)

  ADD_EXECUTABLE(fully-connected-test test/fully-connected.cc)
  SET_TARGET_PROPERTIES(fully-connected-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(fully-connected-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(fully-connected-test PRIVATE pytorch_qnnpack clog cpuinfo fp16 gtest gtest_main)
  ADD_TEST(fully-connected-test fully-connected-test)

  ADD_EXECUTABLE(channel-shuffle-test test/channel-shuffle.cc)
  SET_TARGET_PROPERTIES(channel-shuffle-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(channel-shuffle-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(channel-shuffle-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(channel-shuffle-test channel-shuffle-test)

  ADD_EXECUTABLE(add-test test/add.cc)
  SET_TARGET_PROPERTIES(add-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(add-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(add-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(add-test add-test)

  ADD_EXECUTABLE(leaky-relu-test test/leaky-relu.cc)
  SET_TARGET_PROPERTIES(leaky-relu-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(leaky-relu-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(leaky-relu-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(leaky-relu-test leaky-relu-test)

  ADD_EXECUTABLE(sigmoid-test test/sigmoid.cc)
  SET_TARGET_PROPERTIES(sigmoid-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sigmoid-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(sigmoid-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(sigmoid-test sigmoid-test)

  ADD_EXECUTABLE(clamp-test test/clamp.cc)
  SET_TARGET_PROPERTIES(clamp-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(clamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(clamp-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(clamp-test clamp-test)

  ADD_EXECUTABLE(softargmax-test test/softargmax.cc)
  SET_TARGET_PROPERTIES(softargmax-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(softargmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(softargmax-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(softargmax-test softargmax-test)

  ADD_EXECUTABLE(tanh-test test/tanh.cc)
  SET_TARGET_PROPERTIES(tanh-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(tanh-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(tanh-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(tanh-test tanh-test)

  ADD_EXECUTABLE(max-pooling-test test/max-pooling.cc)
  SET_TARGET_PROPERTIES(max-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(max-pooling-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(max-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(max-pooling-test max-pooling-test)

  ADD_EXECUTABLE(average-pooling-test test/average-pooling.cc)
  SET_TARGET_PROPERTIES(average-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(average-pooling-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(average-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(average-pooling-test average-pooling-test)

  ADD_EXECUTABLE(global-average-pooling-test test/global-average-pooling.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(global-average-pooling-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(global-average-pooling-test PRIVATE pytorch_qnnpack cpuinfo gtest gtest_main)
  ADD_TEST(global-average-pooling-test global-average-pooling-test)

  # ---[ Build unit tests for micro-kernels
  ADD_EXECUTABLE(q8gemm-test test/q8gemm.cc)
  SET_TARGET_PROPERTIES(q8gemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8gemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8gemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8gemm-test q8gemm-test)

  ADD_EXECUTABLE(q8conv-test test/q8conv.cc)
  SET_TARGET_PROPERTIES(q8conv-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8conv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8conv-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8conv-test q8conv-test)

  ADD_EXECUTABLE(q8dwconv-test test/q8dwconv.cc)
  SET_TARGET_PROPERTIES(q8dwconv-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8dwconv-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8dwconv-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8dwconv-test q8dwconv-test)

  ADD_EXECUTABLE(q8vadd-test test/q8vadd.cc)
  SET_TARGET_PROPERTIES(q8vadd-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8vadd-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8vadd-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8vadd-test q8vadd-test)

  ADD_EXECUTABLE(q8avgpool-test test/q8avgpool.cc)
  SET_TARGET_PROPERTIES(q8avgpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8avgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8avgpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8avgpool-test q8avgpool-test)

  ADD_EXECUTABLE(q8gavgpool-test test/q8gavgpool.cc)
  SET_TARGET_PROPERTIES(q8gavgpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8gavgpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(q8gavgpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(q8gavgpool-test q8gavgpool-test)

  ADD_EXECUTABLE(u8maxpool-test test/u8maxpool.cc)
  SET_TARGET_PROPERTIES(u8maxpool-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(u8maxpool-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8maxpool-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8maxpool-test u8maxpool-test)

  ADD_EXECUTABLE(u8clamp-test test/u8clamp.cc)
  SET_TARGET_PROPERTIES(u8clamp-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(u8clamp-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8clamp-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8clamp-test u8clamp-test)

  ADD_EXECUTABLE(u8rmax-test test/u8rmax.cc)
  SET_TARGET_PROPERTIES(u8rmax-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(u8rmax-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8rmax-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8rmax-test u8rmax-test)

  ADD_EXECUTABLE(u8lut32norm-test test/u8lut32norm.cc)
  SET_TARGET_PROPERTIES(u8lut32norm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(u8lut32norm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(u8lut32norm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(u8lut32norm-test u8lut32norm-test)

  ADD_EXECUTABLE(x8lut-test test/x8lut.cc)
  SET_TARGET_PROPERTIES(x8lut-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(x8lut-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x8lut-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x8lut-test x8lut-test)

  ADD_EXECUTABLE(x8zip-test test/x8zip.cc)
  SET_TARGET_PROPERTIES(x8zip-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(x8zip-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(x8zip-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(x8zip-test x8zip-test)

  ADD_EXECUTABLE(hgemm-test test/hgemm.cc)
  SET_TARGET_PROPERTIES(hgemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(hgemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(hgemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(hgemm-test hgemm-test)

  ADD_EXECUTABLE(sgemm-test test/sgemm.cc)
  SET_TARGET_PROPERTIES(sgemm-test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sgemm-test PRIVATE src test)
  TARGET_LINK_LIBRARIES(sgemm-test PRIVATE pytorch_qnnpack cpuinfo fp16 gtest gtest_main)
  ADD_TEST(sgemm-test sgemm-test)
ENDIF()

# ---[ QNNPACK micro-benchmarks
IF(PYTORCH_QNNPACK_BUILD_BENCHMARKS)
  # ---[ Build google benchmark
  IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${GOOGLEBENCHMARK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark")
  ENDIF()

  ADD_EXECUTABLE(add-bench bench/add.cc)
  SET_TARGET_PROPERTIES(add-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(add-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(average-pooling-bench bench/average-pooling.cc)
  SET_TARGET_PROPERTIES(average-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(average-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(channel-shuffle-bench bench/channel-shuffle.cc)
  SET_TARGET_PROPERTIES(channel-shuffle-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(channel-shuffle-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(convolution-bench bench/convolution.cc)
  SET_TARGET_PROPERTIES(convolution-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(convolution-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(global-average-pooling-bench bench/global-average-pooling.cc)
  SET_TARGET_PROPERTIES(global-average-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(global-average-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(max-pooling-bench bench/max-pooling.cc)
  SET_TARGET_PROPERTIES(max-pooling-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(max-pooling-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(sigmoid-bench bench/sigmoid.cc)
  SET_TARGET_PROPERTIES(sigmoid-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(sigmoid-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(softargmax-bench bench/softargmax.cc)
  SET_TARGET_PROPERTIES(softargmax-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(softargmax-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(tanh-bench bench/tanh.cc)
  SET_TARGET_PROPERTIES(tanh-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_LINK_LIBRARIES(tanh-bench PRIVATE pytorch_qnnpack benchmark)

  ADD_EXECUTABLE(q8gemm-bench bench/q8gemm.cc)
  SET_TARGET_PROPERTIES(q8gemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(q8gemm-bench PRIVATE src)
  TARGET_COMPILE_DEFINITIONS(q8gemm-bench PRIVATE pytorch_PYTORCH_QNNPACK_BENCHMARK_GEMMLOWP=0)
  TARGET_LINK_LIBRARIES(q8gemm-bench PRIVATE pytorch_qnnpack cpuinfo fp16 benchmark)

  ADD_EXECUTABLE(hgemm-bench bench/hgemm.cc)
  SET_TARGET_PROPERTIES(hgemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(hgemm-bench PRIVATE src)
  TARGET_LINK_LIBRARIES(hgemm-bench PRIVATE pytorch_qnnpack cpuinfo fp16 benchmark)

  ADD_EXECUTABLE(sgemm-bench bench/sgemm.cc)
  SET_TARGET_PROPERTIES(sgemm-bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  TARGET_INCLUDE_DIRECTORIES(sgemm-bench PRIVATE src)
  TARGET_LINK_LIBRARIES(sgemm-bench PRIVATE pytorch_qnnpack cpuinfo benchmark)
ENDIF()
