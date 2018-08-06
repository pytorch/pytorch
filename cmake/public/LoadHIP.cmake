set(PYTORCH_FOUND_HIP FALSE)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()

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

# HIPBLAS_PATH
IF(NOT DEFINED ENV{HIPBLAS_PATH})
  SET(HIPBLAS_PATH ${ROCM_PATH}/hipblas)
ELSE()
  SET(HIPBLAS_PATH $ENV{HIPBLAS_PATH})
ENDIF()

# ROCBLAS_PATH
IF(NOT DEFINED ENV{ROCBLAS_PATH})
  SET(ROCBLAS_PATH ${ROCM_PATH}/rocblas)
ELSE()
  SET(ROCBLAS_PATH $ENV{ROCBLAS_PATH})
ENDIF()

# HIPSPARSE_PATH
IF(NOT DEFINED ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH ${ROCM_PATH}/hcsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()

# THRUST_PATH
IF(DEFINED ENV{THRUST_PATH})
  SET(THRUST_PATH $ENV{THRUST_PATH})
ELSEIF(DEFINED ENV{THRUST_ROOT})
  # TODO: Remove support of THRUST_ROOT environment variable
  SET(THRUST_PATH $ENV{THRUST_ROOT})
ELSE()
  SET(THRUST_PATH ${ROCM_PATH}/Thrust)
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

# Disable Asserts In Code (Can't use asserts on HIP stack.)
ADD_DEFINITIONS(-DNDEBUG)

# Find the HIP Package
FIND_PACKAGE(HIP 1.0)

IF(HIP_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)

  ### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###
  # https://github.com/ROCm-Developer-Tools/HIP/pull/558 #
  set(CMAKE_SHARED_LIBRARY_SONAME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG})
  set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" )
  set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared" )
  set(CMAKE_HIP_CREATE_SHARED_MODULE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <LINK_LIBRARIES> -shared" )
  set(CMAKE_HIP_ARCHIVE_CREATE ${CMAKE_CXX_ARCHIVE_CREATE})
  set(CMAKE_HIP_ARCHIVE_APPEND ${CMAKE_CXX_ARCHIVE_APPEND})
  set(CMAKE_HIP_ARCHIVE_FINISH ${CMAKE_CXX_ARCHIVE_FINISH})
  SET(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  SET(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  ### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###

  set(rocrand_DIR ${ROCRAND_PATH}/lib/cmake/rocrand)
  set(hiprand_DIR ${HIPRAND_PATH}/lib/cmake/hiprand)
  set(rocblas_DIR ${ROCBLAS_PATH}/lib/cmake/rocblas)
  set(miopen_DIR ${MIOPEN_PATH}/lib/cmake/miopen)
  set(hipblas_DIR ${HIPBLAS_PATH}/lib/cmake/hipblas)
  set(hipsparse_DIR ${HIPSPARSE_PATH}/lib/cmake/hipsparse)

  find_package(rocrand REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(miopen REQUIRED)
  #find_package(hipblas REQUIRED) There's a bug with the CMake file in the Hipblas package.
  #find_package(hipsparse REQUIRED)

  # TODO: hip_hcc has an interface include flag "-hc" which is only
  # recognizable by hcc, but not gcc and clang. Right now in our
  # setup, hcc is only used for linking, but it should be used to
  # compile the *_hip.cc files as well.
  FIND_LIBRARY(PYTORCH_HIP_HCC_LIBRARIES hip_hcc HINTS ${HIP_PATH}/lib)
  # TODO: miopen_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  FIND_LIBRARY(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS ${MIOPEN_PATH}/lib)
  FIND_LIBRARY(hiprand_LIBRARIES hiprand HINTS ${HIPRAND_PATH}/lib)
  FIND_LIBRARY(hipblas_LIBRARIES hipblas HINTS ${HIPBLAS_PATH}/lib)
  FIND_LIBRARY(hipsparse_LIBRARIES hipsparse HINTS ${HIPSPARSE_PATH}/lib)


  # Necessary includes for building PyTorch since we include HIP headers that depend on hcc/hsa headers.
  set(hcc_INCLUDE_DIRS ${HCC_PATH}/include)
  set(hsa_INCLUDE_DIRS ${HSA_PATH}/include)

  set(thrust_INCLUDE_DIRS ${THRUST_PATH} ${THRUST_PATH}/thrust/system/cuda/detail/cub-hip)

ENDIF()
