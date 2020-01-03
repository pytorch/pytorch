# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)

INCLUDE(GNUInstallDirs)

# ---[ Project and semantic versioning.
PROJECT(clog C CXX)

# ---[ Options.
SET(CLOG_RUNTIME_TYPE "default" CACHE STRING "Type of runtime library (shared, static, or default) to use")
SET_PROPERTY(CACHE CLOG_RUNTIME_TYPE PROPERTY STRINGS default static shared)
IF(ANDROID)
  OPTION(CLOG_LOG_TO_STDIO "Log errors, warnings, and information to stdout/stderr" OFF)
ELSE()
  OPTION(CLOG_LOG_TO_STDIO "Log errors, warnings, and information to stdout/stderr" ON)
ENDIF()
OPTION(CLOG_BUILD_TESTS "Build clog tests" ON)

# ---[ CMake options
IF(CLOG_BUILD_TESTS)
  ENABLE_TESTING()
ENDIF()

MACRO(CLOG_TARGET_RUNTIME_LIBRARY target)
  IF(MSVC AND NOT CLOG_RUNTIME_TYPE STREQUAL "default")
    IF(CLOG_RUNTIME_TYPE STREQUAL "shared")
      TARGET_COMPILE_OPTIONS(${target} PRIVATE
        "/MD$<$<CONFIG:Debug>:d>")
    ELSEIF(CLOG_RUNTIME_TYPE STREQUAL "static")
      TARGET_COMPILE_OPTIONS(${target} PRIVATE
        "/MT$<$<CONFIG:Debug>:d>")
    ENDIF()
  ENDIF()
ENDMACRO()

# ---[ Download deps
SET(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

IF(CLOG_BUILD_TESTS)
  IF(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    SET(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
  ENDIF()
ENDIF()

# ---[ clog library
ADD_LIBRARY(clog STATIC src/clog.c)
SET_TARGET_PROPERTIES(clog PROPERTIES
  C_STANDARD 99
  C_EXTENSIONS NO)
CLOG_TARGET_RUNTIME_LIBRARY(clog)
SET_TARGET_PROPERTIES(clog PROPERTIES PUBLIC_HEADER include/clog.h)
TARGET_INCLUDE_DIRECTORIES(clog BEFORE PUBLIC include)
IF(CLOG_LOG_TO_STDIO)
  TARGET_COMPILE_DEFINITIONS(clog PRIVATE CLOG_LOG_TO_STDIO=1)
ELSE()
  TARGET_COMPILE_DEFINITIONS(clog PRIVATE CLOG_LOG_TO_STDIO=0)
ENDIF()
IF(ANDROID AND NOT CLOG_LOG_TO_STDIO)
  TARGET_LINK_LIBRARIES(clog PRIVATE log)
ENDIF()

INSTALL(TARGETS clog
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# ---[ clog tests
IF(CLOG_BUILD_TESTS)
  # ---[ Build google test
  IF(NOT TARGET gtest)
    IF(MSVC AND NOT CLOG_RUNTIME_TYPE STREQUAL "static")
      SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ENDIF()
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
  ENDIF()

  ADD_EXECUTABLE(clog-test test/clog.cc)
  SET_TARGET_PROPERTIES(clog-test PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS NO)
  CLOG_TARGET_RUNTIME_LIBRARY(clog-test)
  TARGET_LINK_LIBRARIES(clog-test PRIVATE clog gtest gtest_main)
  ADD_TEST(clog-test clog-test)
ENDIF()
