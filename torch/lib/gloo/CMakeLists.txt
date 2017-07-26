cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(gloo CXX C)

# Gloo assumes 64-bit and doesn't run builds/tests for anything else.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Gloo can only be built on 64-bit systems.")
endif()

# We want CMake to glob everything every time.
execute_process(COMMAND
  find "${PROJECT_SOURCE_DIR}" -name "CMakeLists.txt" -exec touch {} \;)

# Local CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Build target options
option(BUILD_TEST "Build test binary (requires gtest)" OFF)
option(BUILD_BENCHMARK "Build benchmark binary (requires hiredis)" OFF)

# Option defaults (so they can be overwritten before declaring the option)
set(USE_REDIS_DEFAULT OFF)
set(USE_IBVERBS_DEFAULT OFF)

# Add Redis support if building benchmark
if(BUILD_BENCHMARK)
  set(USE_REDIS_DEFAULT ON)
endif()

# Options
option(USE_REDIS "Support using Redis for rendezvous" ${USE_REDIS_DEFAULT})
option(USE_IBVERBS "Support ibverbs transport" ${USE_IBVERBS_DEFAULT})

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set -- defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Add install targets by default (override from parent project)
set(GLOO_INSTALL ON CACHE BOOL "")
mark_as_advanced(GLOO_INSTALL)

# Build static or shared libraries (override from parent project)
set(GLOO_STATIC_OR_SHARED ${BUILD_SHARED_LIBS} CACHE STRING "")
mark_as_advanced(GLOO_STATIC_OR_SHARED)

# Process dependencies
include(cmake/Dependencies.cmake)

# Use project root as default include directory
include_directories(${PROJECT_SOURCE_DIR})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Recurse into main project directory
add_subdirectory(gloo)
