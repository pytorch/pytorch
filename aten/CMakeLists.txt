cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDA ${CMAKE_MODULE_PATH})
project(ATen)

cmake_policy(SET CMP0012 NEW)

# RPATH stuff
# see https://cmake.org/Wiki/CMake_RPATH_handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# C++11
set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")

# TH stuff
set(Torch_FOUND 1)
add_definitions(-DTH_INDEX_BASE=0)
set(TH_LINK_STYLE STATIC)
add_subdirectory(src/TH)
include_directories(
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THC
  ${CMAKE_CURRENT_BINARY_DIR}/src/TH
  ${CMAKE_CURRENT_BINARY_DIR}/src/THC
  # sparse
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCS
  ${CMAKE_CURRENT_BINARY_DIR}/src/THS
  ${CMAKE_CURRENT_BINARY_DIR}/src/THCS

  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/src)
add_subdirectory(src/THNN)
add_subdirectory(src/THS)

if(NO_CUDA)
  message("disabling CUDA because NO_CUDA is set")
else()
  find_package(CUDA 5.5 REQUIRED)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
  add_subdirectory(src/THCS)
endif()

set(cwrap_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Declarations.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Local.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THNN/generic/THNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN/generic/THCUNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/nn.yaml
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src/THNN
${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN)

add_subdirectory(src/ATen)
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src
${CMAKE_CURRENT_BINARY_DIR}/src/ATen)
add_subdirectory(src/ATen/test)

if(ATEN_NO_CONTRIB)
  message("disable contrib because ATEN_NO_CONTRIB is set")
else()
  add_subdirectory(contrib/data)
  add_subdirectory(contrib/meter)
endif()
