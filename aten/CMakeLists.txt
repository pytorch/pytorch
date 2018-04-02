cmake_minimum_required(VERSION 3.0)
# TODO: Get rid of /usr/lib/x86_64-linux-gnu/
set(CMAKE_MODULE_PATH
  /usr/lib/x86_64-linux-gnu/
  ${CMAKE_MODULE_PATH})
set(CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ ${CMAKE_LIBRARY_PATH})
set(SUBPROJECT 1)
project(ATen)

include(cmake/Common.cmake)

# Top-level build config
############################################

include(LoadCUDA)
include(LoadOpenMP)
include(LoadMAGMA)
include(LoadARM)
include(LoadSSE)
include(LoadBLAS)
include(LoadLAPACK)

#############################################

set(ATen_CPU_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_CUDA_SRCS)
set(ATen_CUDA_INCLUDE)

add_definitions(-DTH_INDEX_BASE=0)
set(TH_LINK_STYLE STATIC)
add_subdirectory(src/avx_mathfun)
add_subdirectory(src/THRTS)
add_subdirectory(src/TH)
include_directories(
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
  set(CUDA_FLAG -n)
  set(AT_CUDA_ENABLED 0)
else()
  set(AT_CUDA_ENABLED 1)
  include_directories(${CUDA_INCLUDE_DIRS})
  add_subdirectory(src/THCRTS)
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
