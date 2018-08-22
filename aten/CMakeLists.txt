if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(ATen CXX C)
  include(CMakeDependentOption)
  option(USE_CUDA "Use CUDA" ON)
  option(USE_ROCM "Use ROCm" OFF)
  option(USE_CUDNN "Use cuDNN" ON)
  option(USE_MKLDNN "Use MKLDNN" ON)
  cmake_dependent_option(
      USE_CUDNN "Use cuDNN" ON
      "USE_CUDA" OFF)
  option(ATEN_NO_TEST "Do not build ATen test binaries" OFF)
else()
  if (BUILD_ATEN_MOBILE)
    return()
  endif()
endif()

# Find modules
list(APPEND CMAKE_MODULE_PATH
  /usr/lib/x86_64-linux-gnu/
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules_CUDA_fix)
list(APPEND CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/)

cmake_policy(SET CMP0012 NEW)

#############################################

set(ATen_CPU_SRCS)
set(ATen_CPU_TEST_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_THIRD_PARTY_INCLUDE)
set(ATen_CUDA_SRCS)
set(ATen_CUDA_TEST_SRCS)
set(ATen_CUDA_INCLUDE)
set(ATen_CPU_DEPENDENCY_LIBS)
set(ATen_CUDA_DEPENDENCY_LIBS)
set(ATen_PUBLIC_CUDA_DEPENDENCY_LIBS)
SET(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
SET(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
SET(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")

if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  # ---[ Build variables set within the cmake tree
  include(../cmake/BuildVariables.cmake)
  set(CAFFE2_WHITELIST "" CACHE STRING "A whitelist file of files that one should build.")

  # ---[ Misc checks to cope with various compiler modes
  include(../cmake/MiscCheck.cmake)

  # External projects
  include(ExternalProject)

  # ---[ Utils
  # TODO: merge the following 3 files into cmake/public/utils.cmake.
  include(../cmake/Utils.cmake)
  include(../cmake/public/utils.cmake)

  # ---[ Dependencies
  include(../cmake/Dependencies.cmake)
  list(APPEND ATen_CPU_INCLUDE ${Caffe2_CPU_INCLUDE})
  list(APPEND ATen_CUDA_INCLUDE ${Caffe2_GPU_INCLUDE})
  list(APPEND ATen_CPU_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS})
  list(APPEND ATen_CUDA_DEPENDENCY_LIBS ${Caffe2_CUDA_DEPENDENCY_LIBS})
  list(APPEND ATen_PUBLIC_CUDA_DEPENDENCY_LIBS
    ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
endif()

if(USE_CUDA)
  list(APPEND ATen_CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
endif()

set(TH_LINK_STYLE STATIC)
add_subdirectory(src/TH)
set(TH_CPU_INCLUDE
  # dense
  ${CMAKE_CURRENT_SOURCE_DIR}/src/TH
  ${CMAKE_CURRENT_BINARY_DIR}/src/TH
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/aten/src)
list(APPEND ATen_CPU_INCLUDE ${TH_CPU_INCLUDE})

if(USE_CUDA OR USE_ROCM)
  set(TH_CUDA_INCLUDE
    # dense
    ${CMAKE_CURRENT_SOURCE_DIR}/src/THC
    ${CMAKE_CURRENT_BINARY_DIR}/src/THC)
  list(APPEND ATen_CUDA_INCLUDE ${TH_CUDA_INCLUDE})
endif()

add_subdirectory(src/THNN)

# Find the HIP package, set the HIP paths, load the HIP CMake.
IF(USE_ROCM)
  include(LoadHIP)
  if (NOT PYTORCH_FOUND_HIP)
    MESSAGE(FATAL_ERROR
      "Could not find HIP installation")
  endif()
ENDIF()

IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
ENDIF(MSVC)

if(USE_ROCM)
  SET(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
  message("ROCm is enabled.")
elseif(USE_CUDA)
  SET(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THC)
  add_subdirectory(src/THCUNN)
else()
  message("disabling CUDA because USE_CUDA is set false")
  SET(AT_CUDA_ENABLED 0)
endif()

list(APPEND ATen_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THNN
  ${CMAKE_CURRENT_SOURCE_DIR}/src/THCUNN)

list(APPEND ATen_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catch/single_include
  ${CMAKE_CURRENT_BINARY_DIR}/src/ATen)
add_subdirectory(src/ATen)

if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  # Pass source, includes, and libs to parent
  set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
  set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} PARENT_SCOPE)
  set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
  set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
  set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
  set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
  set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
  set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
  set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
  set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
endif()
