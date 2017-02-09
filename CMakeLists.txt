cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(gloo CXX C)

# We want CMake to glob everything every time.
execute_process(COMMAND
  find "${PROJECT_SOURCE_DIR}" -name "CMakeLists.txt" -exec touch {} \;)

# Local CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Options
option(USE_REDIS "Support using Redis for rendezvous" OFF)
option(USE_IBVERBS "Support ibverbs transport" OFF)

# Build target options
option(BUILD_TEST "Build test binary (requires gtest)" OFF)
option(BUILD_BENCHMARK "Build benchmark binary (requires hiredis)" OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set -- defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Process dependencies
include(cmake/Dependencies.cmake)

# Use project root as default include directory
include_directories(${PROJECT_SOURCE_DIR})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Recurse into main project directory
add_subdirectory(gloo)
