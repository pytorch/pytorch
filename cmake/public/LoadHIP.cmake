set(PYTORCH_FOUND_HIP FALSE)

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH /opt/rocm)
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()

# HIP_PATH
if(NOT DEFINED ENV{HIP_PATH})
  set(HIP_PATH ${ROCM_PATH}/hip)
else()
  set(HIP_PATH $ENV{HIP_PATH})
endif()

if(NOT EXISTS ${HIP_PATH})
  return()
endif()

# HCC_PATH
if(NOT DEFINED ENV{HCC_PATH})
  set(HCC_PATH ${ROCM_PATH}/hcc)
else()
  set(HCC_PATH $ENV{HCC_PATH})
endif()

# HSA_PATH
if(NOT DEFINED ENV{HSA_PATH})
  set(HSA_PATH ${ROCM_PATH}/hsa)
else()
  set(HSA_PATH $ENV{HSA_PATH})
endif()

# ROCBLAS_PATH
if(NOT DEFINED ENV{ROCBLAS_PATH})
  set(ROCBLAS_PATH ${ROCM_PATH}/rocblas)
else()
  set(ROCBLAS_PATH $ENV{ROCBLAS_PATH})
endif()

# ROCFFT_PATH
if(NOT DEFINED ENV{ROCFFT_PATH})
  set(ROCFFT_PATH ${ROCM_PATH}/rocfft)
else()
  set(ROCFFT_PATH $ENV{ROCFFT_PATH})
endif()

# HIPFFT_PATH
if(NOT DEFINED ENV{HIPFFT_PATH})
  set(HIPFFT_PATH ${ROCM_PATH}/hipfft)
else()
  set(HIPFFT_PATH $ENV{HIPFFT_PATH})
endif()

# HIPSPARSE_PATH
if(NOT DEFINED ENV{HIPSPARSE_PATH})
  set(HIPSPARSE_PATH ${ROCM_PATH}/hipsparse)
else()
  set(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
endif()

# THRUST_PATH
if(DEFINED ENV{THRUST_PATH})
  set(THRUST_PATH $ENV{THRUST_PATH})
else()
  set(THRUST_PATH ${ROCM_PATH}/include)
endif()

# HIPRAND_PATH
if(NOT DEFINED ENV{HIPRAND_PATH})
  set(HIPRAND_PATH ${ROCM_PATH}/hiprand)
else()
  set(HIPRAND_PATH $ENV{HIPRAND_PATH})
endif()

# ROCRAND_PATH
if(NOT DEFINED ENV{ROCRAND_PATH})
  set(ROCRAND_PATH ${ROCM_PATH}/rocrand)
else()
  set(ROCRAND_PATH $ENV{ROCRAND_PATH})
endif()

# MIOPEN_PATH
if(NOT DEFINED ENV{MIOPEN_PATH})
  set(MIOPEN_PATH ${ROCM_PATH}/miopen)
else()
  set(MIOPEN_PATH $ENV{MIOPEN_PATH})
endif()

# RCCL_PATH
if(NOT DEFINED ENV{RCCL_PATH})
  set(RCCL_PATH ${ROCM_PATH}/rccl)
else()
  set(RCCL_PATH $ENV{RCCL_PATH})
endif()

# ROCPRIM_PATH
if(NOT DEFINED ENV{ROCPRIM_PATH})
  set(ROCPRIM_PATH ${ROCM_PATH}/rocprim)
else()
  set(ROCPRIM_PATH $ENV{ROCPRIM_PATH})
endif()

# HIPCUB_PATH
if(NOT DEFINED ENV{HIPCUB_PATH})
  set(HIPCUB_PATH ${ROCM_PATH}/hipcub)
else()
  set(HIPCUB_PATH $ENV{HIPCUB_PATH})
endif()

# ROCTHRUST_PATH
if(NOT DEFINED ENV{ROCTHRUST_PATH})
  set(ROCTHRUST_PATH ${ROCM_PATH}/rocthrust)
else()
  set(ROCTHRUST_PATH $ENV{ROCTHRUST_PATH})
endif()

# ROCTRACER_PATH
if(NOT DEFINED ENV{ROCTRACER_PATH})
  set(ROCTRACER_PATH ${ROCM_PATH}/roctracer)
else()
  set(ROCTRACER_PATH $ENV{ROCTRACER_PATH})
endif()

if(NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
  set(PYTORCH_ROCM_ARCH gfx803;gfx900;gfx906;gfx908)
else()
  set(PYTORCH_ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
endif()

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

# Disable Asserts In Code (Can't use asserts on HIP stack.)
add_definitions(-DNDEBUG)

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
endmacro()

# Find the HIP Package
find_package_and_print_version(HIP 1.0)

if(HIP_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)

  # Find ROCM version for checks
  file(READ "${ROCM_PATH}/.info/version-dev" ROCM_VERSION_DEV_RAW)
  string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)-.*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})
  if(ROCM_VERSION_DEV_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
  endif()
  message("\n***** ROCm version from ${ROCM_PATH}/.info/version-dev ****\n")
  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")

  message("\n***** Library versions from dpkg *****\n")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-libs COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hsakmt-roct COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocr-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep -w hcc COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip_base COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip_hcc COMMAND awk "{print $2 \" VERSION: \" $3}")

  message("\n***** Library versions from cmake find_package *****\n")

  set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  ### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###

  set(hip_DIR ${HIP_PATH}/lib/cmake/hip)
  set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
  set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
  set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
  set(rocrand_DIR ${ROCRAND_PATH}/lib/cmake/rocrand)
  set(hiprand_DIR ${HIPRAND_PATH}/lib/cmake/hiprand)
  set(rocblas_DIR ${ROCBLAS_PATH}/lib/cmake/rocblas)
  set(miopen_DIR ${MIOPEN_PATH}/lib/cmake/miopen)
  set(rocfft_DIR ${ROCFFT_PATH}/lib/cmake/rocfft)
  set(hipfft_DIR ${HIPFFT_PATH}/lib/cmake/hipfft)
  set(hipsparse_DIR ${HIPSPARSE_PATH}/lib/cmake/hipsparse)
  set(rccl_DIR ${RCCL_PATH}/lib/cmake/rccl)
  set(rocprim_DIR ${ROCPRIM_PATH}/lib/cmake/rocprim)
  set(hipcub_DIR ${HIPCUB_PATH}/lib/cmake/hipcub)
  set(rocthrust_DIR ${ROCTHRUST_PATH}/lib/cmake/rocthrust)

  find_package_and_print_version(hip REQUIRED)
  find_package_and_print_version(hsa-runtime64 REQUIRED)
  find_package_and_print_version(amd_comgr REQUIRED)
  find_package_and_print_version(rocrand REQUIRED)
  find_package_and_print_version(hiprand REQUIRED)
  find_package_and_print_version(rocblas REQUIRED)
  find_package_and_print_version(miopen REQUIRED)
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "4.1.0")
    find_package_and_print_version(hipfft REQUIRED)
  else()
    find_package_and_print_version(rocfft REQUIRED)
  endif()
  find_package_and_print_version(hipsparse REQUIRED)
  find_package_and_print_version(rccl)
  find_package_and_print_version(rocprim REQUIRED)
  find_package_and_print_version(hipcub REQUIRED)
  find_package_and_print_version(rocthrust REQUIRED)

  if(HIP_COMPILER STREQUAL clang)
    set(hip_library_name amdhip64)
  else()
    set(hip_library_name hip_hcc)
  endif()
  message("HIP library name: ${hip_library_name}")

  # TODO: hip_hcc has an interface include flag "-hc" which is only
  # recognizable by hcc, but not gcc and clang. Right now in our
  # setup, hcc is only used for linking, but it should be used to
  # compile the *_hip.cc files as well.
  find_library(PYTORCH_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${HIP_PATH}/lib)
  # TODO: miopen_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  find_library(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS ${MIOPEN_PATH}/lib)
  # TODO: rccl_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  find_library(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES} HINTS ${RCCL_PATH}/lib)
  # hiprtc is part of HIP
  find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
  # roctx is part of roctracer
  find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCTRACER_PATH}/lib)
  set(roctracer_INCLUDE_DIRS ${ROCTRACER_PATH}/include)
endif()
