# BSD 3 Clause
# Copyright 2023 Advanced Micro Devices, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fmha_api)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()
if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
  set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
else()
  set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
endif()
# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH ${ROCM_PATH}/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()



IF(NOT EXISTS ${HIP_PATH})
  return()
ENDIF()



# HCC_PATH
IF(NOT DEFINED ENV{HCC_PATH})
  SET(HCC_PATH ${ROCM_PATH}/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()



# HSA_PATH
IF(NOT DEFINED ENV{HSA_PATH})
  SET(HSA_PATH ${ROCM_PATH}/hsa)
ELSE()
  SET(HSA_PATH $ENV{HSA_PATH})
ENDIF()



# ROCBLAS_PATH
IF(NOT DEFINED ENV{ROCBLAS_PATH})
  SET(ROCBLAS_PATH ${ROCM_PATH}/rocblas)
ELSE()
  SET(ROCBLAS_PATH $ENV{ROCBLAS_PATH})
ENDIF()



# ROCSPARSE_PATH
IF(NOT DEFINED ENV{ROCSPARSE_PATH})
  SET(ROCSPARSE_PATH ${ROCM_PATH}/rocsparse)
ELSE()
  SET(ROCSPARSE_PATH $ENV{ROCSPARSE_PATH})
ENDIF()



# ROCFFT_PATH
IF(NOT DEFINED ENV{ROCFFT_PATH})
  SET(ROCFFT_PATH ${ROCM_PATH}/rocfft)
ELSE()
  SET(ROCFFT_PATH $ENV{ROCFFT_PATH})
ENDIF()



# HIPSPARSE_PATH
IF(NOT DEFINED ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH ${ROCM_PATH}/hipsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()



# THRUST_PATH
IF(NOT DEFINED ENV{THRUST_PATH})
  SET(THRUST_PATH ${ROCM_PATH}/include)
ELSE()
  SET(THRUST_PATH $ENV{THRUST_PATH})
ENDIF()



# HIPRAND_PATH
IF(NOT DEFINED ENV{HIPRAND_PATH})
  SET(HIPRAND_PATH ${ROCM_PATH}/hiprand)
ELSE()
  SET(HIPRAND_PATH $ENV{HIPRAND_PATH})
ENDIF()



# ROCRAND_PATH
IF(NOT DEFINED ENV{ROCRAND_PATH})
  SET(ROCRAND_PATH ${ROCM_PATH}/rocrand)
ELSE()
  SET(ROCRAND_PATH $ENV{ROCRAND_PATH})
ENDIF()



# MIOPEN_PATH
IF(NOT DEFINED ENV{MIOPEN_PATH})
  SET(MIOPEN_PATH ${ROCM_PATH}/miopen)
ELSE()
  SET(MIOPEN_PATH $ENV{MIOPEN_PATH})
ENDIF()



# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

find_package(HIP)

set(CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH "/opt/conda/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

find_package(rocblas)
find_package(hipfft)
find_package(hiprand)
find_package(hipsparse)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/include)
include_directories(/opt/conda/include/python3.8)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src FLA_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel/library/src/utility CK_SRCS)

add_executable(fmha_api fmha_api.cpp ${FLA_SRCS} ${CK_SRCS})
target_link_libraries(fmha_api "${TORCH_LIBRARIES}")

message("${TORCH_LIBRARIES}")
