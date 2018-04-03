cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDA
  /usr/lib/x86_64-linux-gnu/
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TH/cmake
  ${CMAKE_MODULE_PATH})
SET(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})
project(ATen)

include(Common)

# Top-level build config
############################################

# Detect CUDA architecture and get best NVCC flags
# finding cuda must be first because other things depend on the result
find_package(CUDA 5.5)
include(CudaCompilerSettings)
include(TestForCudaHalf)

include(LoadOpenMP)
include(LoadMAGMA)
include(LoadARM)
include(LoadSSE)
include(LoadBLAS)

find_package(LAPACK)
if(LAPACK_FOUND)
  set(USE_LAPACK 1) # TODO: eliminate this variable
endif()

#############################################

set(ATen_CPU_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_CUDA_SRCS)
set(ATen_CUDA_INCLUDE)
SET(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
SET(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
SET(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

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
  SET(CUDA_FLAG -n)
  SET(AT_CUDA_ENABLED 0)
else()
  SET(AT_CUDA_ENABLED 1)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  find_package(CUDA 5.5 REQUIRED)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
  add_subdirectory(src/THCS)
endif()

find_package(CuDNN)
IF(NOT AT_CUDA_ENABLED OR NOT CUDNN_FOUND)
  MESSAGE(STATUS "CuDNN not found. Compiling without CuDNN support")
  set(AT_CUDNN_ENABLED 0)
ELSE()
  INCLUDE_DIRECTORIES(BEFORE ${CUDNN_INCLUDE_DIRS})
  set(AT_CUDNN_ENABLED 1)
ENDIF()

if(NO_MKLDNN)
  message("disabling MKLDNN because NO_MKLDNN is set")
  set(AT_MKLDNN_ENABLED 0)
else()
  find_package(MKLDNN)
  if(NOT MKLDNN_FOUND)
    message(STATUS "MKLDNN not found. Compiling without MKLDNN support")
    set(AT_MKLDNN_ENABLED 0)
  else()
    INCLUDE_DIRECTORIES(${MKLDNN_INCLUDE_DIRS})
    set(AT_MKLDNN_ENABLED 1)
  endif()
endif()

set(cwrap_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/Declarations.cwrap
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THNN/generic/THNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN/generic/THCUNN.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/nn.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/native/native_functions.yaml
)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src/THNN
${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN)

add_subdirectory(src/ATen)
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src
${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/utils/catch/single_include
${CMAKE_CURRENT_BINARY_DIR}/src/ATen)
if(NOT NO_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()
add_subdirectory(src/ATen/test)

if(ATEN_NO_CONTRIB)
  message("disable contrib because ATEN_NO_CONTRIB is set")
else()
  add_subdirectory(contrib/data)
  add_subdirectory(contrib/meter)
endif()
