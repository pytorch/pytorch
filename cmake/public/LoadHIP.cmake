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

if(NOT EXISTS ${ROCM_PATH})
  return()
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

  set(CMAKE_HIP_CLANG_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HIP_CLANG_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  ### Remove setting of Flags when FindHIP.CMake PR #558 is accepted.###

  set(hip_DIR ${ROCM_PATH}/lib/cmake/hip)
  set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
  set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
  set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
  set(rocrand_DIR ${ROCM_PATH}/lib/cmake/rocrand)
  set(hiprand_DIR ${ROCM_PATH}/lib/cmake/hiprand)
  set(rocblas_DIR ${ROCM_PATH}/lib/cmake/rocblas)
  set(hipblas_DIR ${ROCM_PATH}/lib/cmake/hipblas)
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
  set(hiprtc_DIR ${ROCM_PATH}/lib/cmake/hiprtc)


  find_package_and_print_version(hip REQUIRED)
  find_package_and_print_version(hsa-runtime64 REQUIRED)
  find_package_and_print_version(amd_comgr REQUIRED)
  find_package_and_print_version(rocrand REQUIRED)
  find_package_and_print_version(hiprand REQUIRED)
  find_package_and_print_version(rocblas REQUIRED)
  find_package_and_print_version(hipblas REQUIRED)
  find_package_and_print_version(hipblaslt REQUIRED)
  find_package_and_print_version(miopen REQUIRED)
  find_package_and_print_version(hipfft REQUIRED)
  find_package_and_print_version(hipsparse REQUIRED)
  find_package_and_print_version(rccl)
  find_package_and_print_version(rocprim REQUIRED)
  find_package_and_print_version(hipcub REQUIRED)
  find_package_and_print_version(rocthrust REQUIRED)
  find_package_and_print_version(hipsolver REQUIRED)
  find_package_and_print_version(hiprtc REQUIRED)


  find_library(PYTORCH_HIP_LIBRARIES amdhip64 HINTS ${ROCM_PATH}/lib)
  # TODO: miopen_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  if(TARGET ${miopen_LIBRARIES})
    set(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES})
  else()
    find_library(PYTORCH_MIOPEN_LIBRARIES ${miopen_LIBRARIES} HINTS ${ROCM_PATH}/lib)
  endif()
  # TODO: rccl_LIBRARIES should return fullpath to the library file,
  # however currently it's just the lib name
  if(TARGET ${rccl_LIBRARIES})
    set(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES})
  else()
    find_library(PYTORCH_RCCL_LIBRARIES ${rccl_LIBRARIES} HINTS ${ROCM_PATH}/lib)
  endif()
  find_library(ROCM_HIPRTC_LIB hiprtc HINTS ${ROCM_PATH}/lib)
  # roctx is part of roctracer
  find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCM_PATH}/lib)

  # check whether HIP declares new types
  set(file "${PROJECT_BINARY_DIR}/hip_new_types.cc")
  file(WRITE ${file} ""
    "#include <hip/library_types.h>\n"
    "int main() {\n"
    "    hipDataType baz = HIP_R_8F_E4M3_FNUZ;\n"
    "    return 0;\n"
    "}\n"
    )

  try_compile(hip_compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ROCM_INCLUDE_DIRS}"
    COMPILE_DEFINITIONS -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
    OUTPUT_VARIABLE hip_compile_output)

  if(hip_compile_result)
    set(HIP_NEW_TYPE_ENUMS ON)
    #message("HIP is using new type enums: ${hip_compile_output}")
    message("HIP is using new type enums")
  else()
    set(HIP_NEW_TYPE_ENUMS OFF)
    #message("HIP is NOT using new type enums: ${hip_compile_output}")
    message("HIP is NOT using new type enums")
  endif()

endif()
