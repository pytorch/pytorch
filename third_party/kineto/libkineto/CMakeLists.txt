# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(kineto VERSION 0.1 LANGUAGES CXX C)

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from libkineto_defs.bzl file
find_package(Python3 COMPONENTS Interpreter)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "exec(open('libkineto_defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

set(KINETO_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (default, static or shared) to build")
set_property(CACHE KINETO_LIBRARY_TYPE PROPERTY STRINGS default shared static)
option(KINETO_BUILD_TESTS "Build kineto unit tests" ON)
option(KINETO_BUILD_BENCHMARKS "Build kineto benchmarks" OFF)

set(LIBKINETO_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBKINETO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBKINETO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBKINETO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBKINETO_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#We should default to a Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if(DEFINED CUDA_SOURCE_DIR AND NOT DEFINED CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${CUDA_SOURCE_DIR}")
  message(STATUS " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

if(NOT ROCM_SOURCE_DIR)
  set(ROCM_SOURCE_DIR "$ENV{ROCM_SOURCE_DIR}")
  message(STATUS " ROCM_SOURCE_DIR = ${ROCM_SOURCE_DIR}")
endif()

# --- Backend selection ---
# KINETO_BACKEND selects the GPU backend (or "cpu" for no GPU profiling).
# Selection is explicit: if the chosen toolkit isn't available, configuration
# fails. Use KINETO_BACKEND=cpu to build without GPU support.

# Track whether the user explicitly set KINETO_BACKEND before the default
# fires below, so we can warn when we fall back to cpu.
set(_kineto_backend_explicit ON)
if(NOT DEFINED CACHE{KINETO_BACKEND} AND NOT DEFINED KINETO_BACKEND)
  set(_kineto_backend_explicit OFF)
endif()

set(KINETO_BACKEND "cpu" CACHE STRING
    "Kineto GPU backend: cpu, cuda, rocm, or xpu")
set_property(CACHE KINETO_BACKEND PROPERTY STRINGS cpu cuda rocm xpu)

set(_KINETO_BACKEND_VALID cpu cuda rocm xpu)
if(NOT KINETO_BACKEND IN_LIST _KINETO_BACKEND_VALID)
  message(FATAL_ERROR
    "KINETO_BACKEND=${KINETO_BACKEND} is invalid. "
    "Must be one of: ${_KINETO_BACKEND_VALID}")
endif()
unset(_KINETO_BACKEND_VALID)
message(STATUS "KINETO_BACKEND = ${KINETO_BACKEND}")

if(NOT _kineto_backend_explicit)
  message(WARNING
    "KINETO_BACKEND was not set; defaulting to 'cpu'. "
    "Pass -DKINETO_BACKEND={cpu,cuda,rocm,xpu} explicitly to silence this warning.")
endif()
unset(_kineto_backend_explicit)

# Per-backend configure-time setup.
if(KINETO_BACKEND STREQUAL "cuda")
  # CUDA / CUPTI detection
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET CUDA::cupti)
    message(FATAL_ERROR
      "KINETO_BACKEND=cuda but CUPTI was not found. "
      "Install the CUDA Toolkit with CUPTI, set CUDAToolkit_ROOT, "
      "or set KINETO_BACKEND=cpu.")
  endif()
  message(STATUS "Found CUPTI")
elseif(KINETO_BACKEND STREQUAL "rocm")
  # ROCm prerequisites and version detection
  if(NOT ROCM_SOURCE_DIR)
    message(FATAL_ERROR
      "KINETO_BACKEND=rocm but ROCM_SOURCE_DIR is not set. "
      "Set ROCM_SOURCE_DIR (or the ROCM_SOURCE_DIR environment variable) "
      "or set KINETO_BACKEND=cpu.")
  endif()
  if(NOT ROCM_INCLUDE_DIRS)
    set(ROCM_INCLUDE_DIRS "${ROCM_SOURCE_DIR}/include")
  endif()

  find_file(ROCM_VERSION_HEADER_PATH
    NAMES rocm-core/rocm_version.h
    NO_DEFAULT_PATH
    PATHS ${ROCM_INCLUDE_DIRS}
  )
  if(NOT EXISTS ${ROCM_VERSION_HEADER_PATH})
    message(FATAL_ERROR
      "KINETO_BACKEND=rocm but rocm-core/rocm_version.h was not found in "
      "${ROCM_INCLUDE_DIRS}. Check ROCM_SOURCE_DIR / ROCM_INCLUDE_DIRS or "
      "set KINETO_BACKEND=cpu.")
  endif()
  set(ROCM_HEADER_FILE ${ROCM_VERSION_HEADER_PATH})

  message(STATUS "Reading ROCM version from: ${ROCM_HEADER_FILE}")
  file(READ "${ROCM_HEADER_FILE}" ROCM_HEADER_CONTENT)

  string(REGEX MATCH "ROCM_VERSION_MAJOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_MAJOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MAJOR)
  string(REGEX MATCH "ROCM_VERSION_MINOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_MINOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MINOR)
  string(REGEX MATCH "ROCM_VERSION_PATCH[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_PATCH" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_PATCH)

  message(STATUS "ROCM major: ${ROCM_VERSION_DEV_MAJOR}")
  message(STATUS "ROCM minor: ${ROCM_VERSION_DEV_MINOR}")
  message(STATUS "ROCM patch: ${ROCM_VERSION_DEV_PATCH}")

  # Kineto uses rocprofiler-sdk, which is supported on ROCm 7.0 and later.
  # roctracer was the only pre-7.0 path and has been removed, so older ROCm
  # versions are no longer supported.
  if(ROCM_VERSION_DEV_MAJOR LESS 7)
    message(FATAL_ERROR
      "KINETO_BACKEND=rocm requires ROCm >= 7.0 (rocprofiler-sdk); detected "
      "ROCm ${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}. "
      "Upgrade ROCm or set KINETO_BACKEND=cpu.")
  endif()
  message(STATUS "Building with: rocprofiler-sdk")
elseif(KINETO_BACKEND STREQUAL "xpu")
  # XPU plugin
  add_subdirectory(src/plugin/xpupti)
endif()

# Define file lists. Backends are mutually exclusive — exactly one (or none)
# is active per build, gated on KINETO_BACKEND.
if(KINETO_BACKEND STREQUAL "cpu")
  get_filelist("get_libkineto_cpu_only_srcs(with_api=False)" LIBKINETO_SRCS)
  message(STATUS " Building CPU-only profilers")
elseif(KINETO_BACKEND STREQUAL "cuda")
  get_filelist("get_libkineto_cupti_srcs(with_api=False)" LIBKINETO_SRCS)
  message(STATUS " Building with cupti")
elseif(KINETO_BACKEND STREQUAL "rocm")
  get_filelist("get_libkineto_rocprofiler_srcs(with_api=False)" LIBKINETO_SRCS)
  message(STATUS " Building with rocprofiler-sdk")
elseif(KINETO_BACKEND STREQUAL "xpu")
  get_filelist("get_libkineto_xpupti_srcs(with_api=False)" LIBKINETO_SRCS)
  message(STATUS " Building with xpupti")
endif()
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
get_filelist("get_libkineto_api_srcs()" LIBKINETO_API_SRCS)

add_library(kineto_base OBJECT ${LIBKINETO_SRCS})
add_library(kineto_api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on libkineto_defs.bzl
add_custom_target(libkineto_defs.bzl DEPENDS libkineto_defs.bzl)
add_dependencies(kineto_base libkineto_defs.bzl)

set(KINETO_DEFINITIONS "KINETO_NAMESPACE=libkineto")
list(APPEND KINETO_DEFINITIONS "ENABLE_IPC_FABRIC")
set(KINETO_COMPILE_OPTIONS)
if(MSVC)
  list(APPEND KINETO_COMPILE_OPTIONS "/utf-8")
endif()
if(KINETO_BACKEND STREQUAL "cuda")
  list(APPEND KINETO_DEFINITIONS "HAS_CUPTI")
elseif(KINETO_BACKEND STREQUAL "rocm")
  list(APPEND KINETO_DEFINITIONS "HAS_ROCTRACER")
  target_compile_definitions(kineto_base PRIVATE "__HIP_PLATFORM_HCC__")
  target_compile_definitions(kineto_base PRIVATE "__HIP_PLATFORM_AMD__")
elseif(KINETO_BACKEND STREQUAL "xpu")
  if(MSVC)
    # MSVC reports __cplusplus as 199711L unless /Zc:__cplusplus is set, which
    # breaks SYCL headers that static_assert __cplusplus >= 201703L.
    list(APPEND KINETO_COMPILE_OPTIONS "/Zc:__cplusplus")
  endif()

  list(APPEND KINETO_COMPILE_OPTIONS ${XPUPTI_BUILD_FLAG})
  if(KINETO_BUILD_TESTS)
    set_target_properties(kineto_base PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(kineto_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

target_compile_definitions(kineto_base PUBLIC "${KINETO_DEFINITIONS}")
target_compile_options(kineto_base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_definitions(kineto_api PUBLIC "${KINETO_DEFINITIONS}")
target_compile_options(kineto_api PRIVATE "${KINETO_COMPILE_OPTIONS}")

# dynolog's ipcfabric is header-only and vendored under third_party/dynolog_headers
# (pinned in version.txt), so only its include paths are needed below -- there
# is no library to build or link.
set(DYNOLOG_INCLUDE_DIR "${LIBKINETO_THIRDPARTY_DIR}/dynolog_headers/")
set(IPCFABRIC_INCLUDE_DIR "${DYNOLOG_INCLUDE_DIR}/dynolog/src/ipcfabric/")

if(NOT TARGET fmt::fmt-header-only)
  if(NOT FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  # Build FMT.
  # FMT and some other libraries use BUILD_SHARED_LIBS to control
  # the library type.
  # Save and restore the value after configuring FMT
  set(FMT_INSTALL OFF)
  add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_BINARY_DIR}/fmt")
  message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
endif()

message(STATUS " DYNOLOG_INCLUDE_DIR = ${DYNOLOG_INCLUDE_DIR}")
message(STATUS " IPCFABRIC_INCLUDE_DIR = ${IPCFABRIC_INCLUDE_DIR}")

target_include_directories(kineto_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>)
target_include_directories(kineto_base SYSTEM PUBLIC
      $<BUILD_INTERFACE:${DYNOLOG_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${IPCFABRIC_INCLUDE_DIR}>)
target_link_libraries(kineto_base PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

target_include_directories(kineto_api PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>)
target_link_libraries(kineto_api PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

if(KINETO_LIBRARY_TYPE STREQUAL "static")
  set(BUILD_SHARED_LIBS OFF)
elseif(KINETO_LIBRARY_TYPE STREQUAL "shared")
  set(BUILD_SHARED_LIBS ON)
  set(WINDOWS_EXPORT_ALL_SYMBOLS ON)
elseif(NOT KINETO_LIBRARY_TYPE STREQUAL "default")
  message(FATAL_ERROR "Unsupported library type ${KINETO_LIBRARY_TYPE}")
endif()

add_library(kineto $<TARGET_OBJECTS:kineto_base> $<TARGET_OBJECTS:kineto_api>)
if(BUILD_SHARED_LIBS)
  set_property(TARGET kineto_base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET kineto_api PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(kineto PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# Strict, pedantic compilation with warnings-as-errors.
# Only enabled for platforms covered by Kineto's CI (CPU-only, CUDA, ROCm on
# Linux). Disabled for Windows, macOS, and XPU builds which lack CI coverage.
if(NOT WIN32 AND NOT APPLE AND NOT KINETO_BACKEND STREQUAL "xpu")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")
endif()

set_target_properties(kineto kineto_base kineto_api PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

target_include_directories(kineto PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>)

# Attach backend-specific includes and libraries — at most one backend active.
# Each backend dependency is attached to BOTH `kineto_base` (the OBJECT
# library, so its own .cpp files compile against the dependency's headers)
# AND `kineto` (the wrapper library, so downstream consumers link the
# dependency). This dual-attach is required because $<TARGET_OBJECTS:> does
# not propagate INTERFACE properties from kineto_base to kineto.
if(KINETO_BACKEND STREQUAL "cuda")
  target_link_libraries(kineto PUBLIC CUDA::cupti CUDA::cudart)
  target_link_libraries(kineto_base PUBLIC CUDA::cupti CUDA::cudart)
elseif(KINETO_BACKEND STREQUAL "rocm")
  target_include_directories(kineto_base SYSTEM PUBLIC
        $<BUILD_INTERFACE:${ROCM_INCLUDE_DIRS}>)
  find_library(ROCPROF_LIBRARY NAMES librocprofiler-sdk.so HINTS
    ${ROCM_SOURCE_DIR}/lib)
  target_link_libraries(kineto "${ROCPROF_LIBRARY}")
  target_link_libraries(kineto_base PUBLIC "${ROCPROF_LIBRARY}")
  find_library(KINETO_HIP_LIBRARY NAMES libamdhip64.so HINTS
    ${ROCM_SOURCE_DIR}/lib)
  target_link_libraries(kineto "${KINETO_HIP_LIBRARY}")
  target_link_libraries(kineto_base PUBLIC "${KINETO_HIP_LIBRARY}")
elseif(KINETO_BACKEND STREQUAL "xpu")
  # Bare form, not $<BUILD_INTERFACE:...>, because XPUPTI_INCLUDE_DIR is a
  # multi-path semicolon list (SYCL + PTI). Wrapping a list in a single
  # generator expression mangles paths after the first. kineto_base is not
  # installed, so the install-interface concern that motivates BUILD_INTERFACE
  # elsewhere doesn't apply here.
  target_include_directories(kineto_base SYSTEM PUBLIC ${XPUPTI_INCLUDE_DIR})
  target_link_libraries(kineto PRIVATE "${XPU_xpupti_LIBRARY}")
endif()
target_compile_definitions(kineto PUBLIC "${KINETO_DEFINITIONS}")

if(KINETO_BUILD_TESTS)
  if(NOT TARGET gtest)
    set(INSTALL_GTEST OFF)
    set(TMP_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory("${LIBKINETO_THIRDPARTY_DIR}/googletest")
    set(BUILD_SHARED_LIBS "${TMP_BUILD_SHARED_LIBS}")
    if(KINETO_BACKEND STREQUAL "xpu")
      set_property(TARGET gtest PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()
  if(NOT TARGET nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    add_subdirectory(
      "${LIBKINETO_THIRDPARTY_DIR}/json"
      "${LIBKINETO_BINARY_DIR}/json")
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

if(KINETO_BUILD_BENCHMARKS)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks"
    "${CMAKE_CURRENT_BINARY_DIR}/benchmarks"
  )
endif()

install(TARGETS kineto
  EXPORT kinetoLibraryConfig
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBKINETO_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kineto")

install(EXPORT kinetoLibraryConfig DESTINATION share/cmake/kineto
  FILE kinetoLibraryConfig.cmake)
