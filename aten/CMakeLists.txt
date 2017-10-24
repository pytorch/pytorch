cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDA ${CMAKE_MODULE_PATH})
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
set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")

# TH stuff
set(Torch_FOUND 1)
add_definitions(-DTH_INDEX_BASE=0)
add_subdirectory(lib/TH)
include_directories(
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/THC
  ${CMAKE_CURRENT_BINARY_DIR}/lib/TH
  ${CMAKE_CURRENT_BINARY_DIR}/lib/THC
  # sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/THS
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/THCS
  ${CMAKE_CURRENT_BINARY_DIR}/lib/THS
  ${CMAKE_CURRENT_BINARY_DIR}/lib/THCS

  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${CMAKE_CURRENT_BINARY_DIR}/lib)
add_subdirectory(lib/THNN)
add_subdirectory(lib/THS)

if(NO_CUDA)
  message("disabling CUDA because NO_CUDA is set")
else()
  find_package(CUDA 5.5 REQUIRED)
  add_subdirectory(lib/THC)
  add_subdirectory(lib/THCUNN)
  add_subdirectory(lib/THCS)
endif()

set(cwrap_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Declarations.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Local.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/THNN/generic/THNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/THCUNN/generic/THCUNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/nn.yaml
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/lib/THNN
${CMAKE_CURRENT_SOURCE_DIR}/lib/THCUNN)

add_subdirectory(src/ATen)
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src
${CMAKE_CURRENT_BINARY_DIR}/src/ATen)
add_subdirectory(src/ATen/test)
add_subdirectory(src/data)
add_subdirectory(src/meter)
