cmake_minimum_required(VERSION 3.0)
project(ATen)

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# C++11
if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

# Nice trace in debug mode
# see https://github.com/bombela/backward-cpp
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_subdirectory(lib/backtrace)
endif()

# TH stuff
set(Torch_FOUND 1)
add_definitions(-DTH_INDEX_BASE=0)
add_subdirectory(lib/TH)
include_directories(
  # dense
  ${CMAKE_SOURCE_DIR}/lib/TH
  ${CMAKE_SOURCE_DIR}/lib/THC
  ${CMAKE_BINARY_DIR}/lib/TH
  ${CMAKE_BINARY_DIR}/lib/THC
  # sparse
  ${CMAKE_SOURCE_DIR}/lib/THS
  ${CMAKE_SOURCE_DIR}/lib/THCS
  ${CMAKE_BINARY_DIR}/lib/THS
  ${CMAKE_BINARY_DIR}/lib/THCS

  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_BINARY_DIR}/lib)
add_subdirectory(lib/THNN)
add_subdirectory(lib/THS)

if(NO_CUDA)
  message("disabling CUDA because NO_CUDA is set")
else()
  find_package(CUDA 5.5)
  if(CUDA_FOUND)
    add_subdirectory(lib/THC)
    add_subdirectory(lib/THCUNN)
    add_subdirectory(lib/THCS)
  else()
    message(STATUS "CUDA not found: disabling THC")
  endif()
endif()

set(cwrap_files
  ${CMAKE_SOURCE_DIR}/tools/Declarations.cwrap
  ${CMAKE_SOURCE_DIR}/src/ATen/Local.cwrap
  ${CMAKE_SOURCE_DIR}/lib/THNN/generic/THNN.h
  ${CMAKE_SOURCE_DIR}/lib/THCUNN/generic/THCUNN.h
)

include_directories(
${CMAKE_SOURCE_DIR}/lib/THNN
${CMAKE_SOURCE_DIR}/lib/THCUNN)

add_subdirectory(src/ATen)
include_directories(
${CMAKE_SOURCE_DIR}/src
${CMAKE_BINARY_DIR}/src/ATen)

add_subdirectory(src/data)
add_subdirectory(src/meter)
