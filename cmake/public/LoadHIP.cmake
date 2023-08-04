set(PYTORCH_FOUND_HIP FALSE)

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH /opt/rocm)
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()
if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
  set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
else()
  set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
endif()
# HIP_PATH
if(NOT DEFINED ENV{HIP_PATH})
  set(HIP_PATH ${ROCM_PATH}/hip)
else()
  set(HIP_PATH $ENV{HIP_PATH})
endif()

#if(NOT EXISTS ${HIP_PATH})
#  return()
#endif()

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

# HIPSOLVER_PATH
if(NOT DEFINED ENV{HIPSOLVER_PATH})
  set(HIPSOLVER_PATH ${ROCM_PATH}/hipsolver)
else()
  set(HIPSOLVER_PATH $ENV{HIPSOLVER_PATH})
endif()

# ROCTRACER_PATH
if(NOT DEFINED ENV{ROCTRACER_PATH})
  set(ROCTRACER_PATH ${ROCM_PATH}/roctracer)
else()
  set(ROCTRACER_PATH $ENV{ROCTRACER_PATH})
endif()

# MAGMA_HOME
if(NOT DEFINED ENV{MAGMA_HOME})
  set(MAGMA_HOME ${ROCM_PATH}/magma)
  set(ENV{MAGMA_HOME} ${ROCM_PATH}/magma)
else()
  set(MAGMA_HOME $ENV{MAGMA_HOME})
endif()

torch_hip_get_arch_list(PYTORCH_ROCM_ARCH)
if(PYTORCH_ROCM_ARCH STREQUAL "")
  message(FATAL_ERROR "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH environment variable to specify GPU archs to build for.")
endif()
message("Building PyTorch for GPU arch: ${PYTORCH_ROCM_ARCH}")

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${CMAKE_MODULE_PATH})

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
endmacro()

# Find the HIP Package
find_package_and_print_version(HIP 1.0)

if(HIP_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)
  set(FOUND_ROCM_VERSION_H FALSE)

  if(EXISTS ${ROCM_PATH}/.info/version-dev)
    # ROCM < 4.5, we don't have the header api file, use flat file
    file(READ "${ROCM_PATH}/.info/version-dev" ROCM_VERSION_DEV_RAW)
    message("\n***** ROCm version from ${ROCM_PATH}/.info/version-dev ****\n")
  endif()

  set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
  set(file "${PROJECT_BINARY_DIR}/detect_rocm_version.cc")

  # Find ROCM version for checks
  # ROCM 5.0 and later will have header api for version management
  if(EXISTS ${ROCM_INCLUDE_DIRS}/rocm_version.h)
    set(FOUND_ROCM_VERSION_H TRUE)
    file(WRITE ${file} ""
      "#include <rocm_version.h>\n"
      )
  elseif(EXISTS ${ROCM_INCLUDE_DIRS}/rocm-core/rocm_version.h)
    set(FOUND_ROCM_VERSION_H TRUE)
    file(WRITE ${file} ""
      "#include <rocm-core/rocm_version.h>\n"
      )
  else()
    message("********************* rocm_version.h couldnt be found ******************\n")
  endif()

  if(FOUND_ROCM_VERSION_H)
    file(APPEND ${file} ""
      "#include <cstdio>\n"

      "#ifndef ROCM_VERSION_PATCH\n"
      "#define ROCM_VERSION_PATCH 0\n"
      "#endif\n"
      "#define STRINGIFYHELPER(x) #x\n"
      "#define STRINGIFY(x) STRINGIFYHELPER(x)\n"
      "int main() {\n"
      "  printf(\"%d.%d.%s\", ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, STRINGIFY(ROCM_VERSION_PATCH));\n"
      "  return 0;\n"
      "}\n"
      )

    try_run(run_result compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ROCM_INCLUDE_DIRS}"
      RUN_OUTPUT_VARIABLE rocm_version_from_header
      COMPILE_OUTPUT_VARIABLE output_var
      )
    # We expect the compile to be successful if the include directory exists.
    if(NOT compile_result)
      message(FATAL_ERROR "Caffe2: Couldn't determine version from header: " ${output_var})
    endif()
    message(STATUS "Caffe2: Header version is: " ${rocm_version_from_header})
    set(ROCM_VERSION_DEV_RAW ${rocm_version_from_header})
    message("\n***** ROCm version from rocm_version.h ****\n")
  endif()

  string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+).*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})

  if(ROCM_VERSION_DEV_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
    math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
  endif()

  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message("ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")

  math(EXPR TORCH_HIP_VERSION "(${HIP_VERSION_MAJOR} * 100) + ${HIP_VERSION_MINOR}")
  message("HIP_VERSION_MAJOR: ${HIP_VERSION_MAJOR}")
  message("HIP_VERSION_MINOR: ${HIP_VERSION_MINOR}")
  message("TORCH_HIP_VERSION: ${TORCH_HIP_VERSION}")

  message("\n***** Library versions from dpkg *****\n")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-libs COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hsakmt-roct COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocr-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep -w hcc COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip-base COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip_hcc COMMAND awk "{print $2 \" VERSION: \" $3}")

  message("\n***** Library versions from cmake find_package *****\n")

  set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  ### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###

  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "6.0.0")
    set(HIP_PATH ${ROCM_PATH})
    set(ROCRAND_PATH ${ROCM_PATH})
    set(HIPRAND_PATH ${ROCM_PATH})
    set(ROCBLAS_PATH ${ROCM_PATH})
    set(MIOPEN_PATH ${ROCM_PATH})
    set(ROCFFT_PATH ${ROCM_PATH})
    set(HIPFFT_PATH ${ROCM_PATH})
    set(HIPSPARSE_PATH ${ROCM_PATH})
    set(RCCL_PATH ${ROCM_PATH})
    set(ROCPRIM_PATH ${ROCM_PATH})
    set(HIPCUB_PATH ${ROCM_PATH})
    set(ROCTHRUST_PATH ${ROCM_PATH})
    set(HIPSOLVER_PATH ${ROCM_PATH})
    set(ROCTRACER_PATH ${ROCM_PATH})
  endif()

  # As of ROCm 5.1.x, all *.cmake files are under /opt/rocm/lib/cmake/<package>
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "5.1.0")
    set(hip_DIR ${ROCM_PATH}/lib/cmake/hip)
    set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
    set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
    set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
    set(rocrand_DIR ${ROCM_PATH}/lib/cmake/rocrand)
    set(hiprand_DIR ${ROCM_PATH}/lib/cmake/hiprand)
    set(rocblas_DIR ${ROCM_PATH}/lib/cmake/rocblas)
    set(hipblaslt_DIR ${ROCM_PATH}/lib/cmake/hipblaslt)
    set(miopen_DIR ${ROCM_PATH}/lib/cmake/miopen)
    set(rocfft_DIR ${ROCM_PATH}/lib/cmake/rocfft)
    set(hipfft_DIR ${ROCM_PATH}/lib/cmake/hipfft)
    set(hipsparse_DIR ${ROCM_PATH}/lib/cmake/hipsparse)
    set(rccl_DIR ${ROCM_PATH}/lib/cmake/rccl)
    set(rocprim_DIR ${ROCM_PATH}/lib/cmake/rocprim)
    set(hipcub_DIR ${ROCM_PATH}/lib/cmake/hipcub)
    set(rocthrust_DIR ${ROCM_PATH}/lib/cmake/rocthrust)
    set(hipsolver_DIR ${ROCM_PATH}/lib/cmake/hipsolver)
  else()
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
    set(hipsolver_DIR ${HIPSOLVER_PATH}/lib/cmake/hipsolver)
  endif()


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
  find_package_and_print_version(hipsolver REQUIRED)

  # Enabling HIP language support
  enable_language(HIP)

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
  if(TARGET ${miopen_LIBRARIES})
    set(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES})
  else()
    find_library(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS ${MIOPEN_PATH}/lib)
  endif()
  # TODO: rccl_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  if(TARGET ${rccl_LIBRARIES})
    set(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES})
  else()
    find_library(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES} HINTS ${RCCL_PATH}/lib)
  endif()
  # hiprtc is part of HIP
  find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
  # roctx is part of roctracer
  find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCTRACER_PATH}/lib)
  # hipblaslt doesn't yet have a cmake package
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "5.6.0")
    find_library(PYTORCH_HIPBLASLT_LIBRARIES hipblaslt HINTS ${ROCM_PATH}/lib)
  endif()
endif()
