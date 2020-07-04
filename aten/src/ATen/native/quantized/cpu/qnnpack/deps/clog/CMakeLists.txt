# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

include(GNUInstallDirs)

# ---[ Project and semantic versioning.
project(clog C CXX)

# ---[ Options.
set(CLOG_RUNTIME_TYPE "default" CACHE STRING "Type of runtime library (shared, static, or default) to use")
set_property(CACHE CLOG_RUNTIME_TYPE PROPERTY STRINGS default static shared)
if(ANDROID)
  option(CLOG_LOG_TO_STDIO "Log errors, warnings, and information to stdout/stderr" OFF)
else()
  option(CLOG_LOG_TO_STDIO "Log errors, warnings, and information to stdout/stderr" ON)
endif()
option(CLOG_BUILD_TESTS "Build clog tests" ON)

# ---[ CMake options
if(CLOG_BUILD_TESTS)
  enable_testing()
endif()

macro(CLOG_TARGET_RUNTIME_LIBRARY target)
  if(MSVC AND NOT CLOG_RUNTIME_TYPE STREQUAL "default")
    if(CLOG_RUNTIME_TYPE STREQUAL "shared")
      target_compile_options(${target} PRIVATE
        "/MD$<$<CONFIG:Debug>:d>")
    elseif(CLOG_RUNTIME_TYPE STREQUAL "static")
      target_compile_options(${target} PRIVATE
        "/MT$<$<CONFIG:Debug>:d>")
    endif()
  endif()
endmacro()

# ---[ Download deps
set(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

if(CLOG_BUILD_TESTS)
  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    message(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
    configure_file(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    set(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
  endif()
endif()

# ---[ clog library
add_library(clog STATIC src/clog.c)
set_target_properties(clog PROPERTIES
  C_STANDARD 99
  C_EXTENSIONS NO)
CLOG_TARGET_RUNTIME_LIBRARY(clog)
set_target_properties(clog PROPERTIES PUBLIC_HEADER include/clog.h)
target_include_directories(clog BEFORE PUBLIC include)
if(CLOG_LOG_TO_STDIO)
  target_compile_definitions(clog PRIVATE CLOG_LOG_TO_STDIO=1)
else()
  target_compile_definitions(clog PRIVATE CLOG_LOG_TO_STDIO=0)
endif()
if(ANDROID AND NOT CLOG_LOG_TO_STDIO)
  target_link_libraries(clog PRIVATE log)
endif()

install(TARGETS clog
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# ---[ clog tests
if(CLOG_BUILD_TESTS)
  # ---[ Build google test
  if(NOT TARGET gtest)
    if(MSVC AND NOT CLOG_RUNTIME_TYPE STREQUAL "static")
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  endif()

  add_executable(clog-test test/clog.cc)
  set_target_properties(clog-test PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS NO)
  CLOG_TARGET_RUNTIME_LIBRARY(clog-test)
  target_link_libraries(clog-test PRIVATE clog gtest gtest_main)
  add_test(clog-test clog-test)
endif()
