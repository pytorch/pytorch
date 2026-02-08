set(PYTORCH_FOUND_HIP FALSE)

# If ROCM_PATH is set, assume intention is to compile with
# ROCm support and error out if the ROCM_PATH does not exist.
# Else ROCM_PATH does not exist, assume a default of /opt/rocm
# In the latter case, if /opt/rocm does not exist emit status
# message and return.
if(DEFINED ENV{ROCM_PATH})
  file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
  if(NOT EXISTS ${ROCM_PATH})
    message(FATAL_ERROR
      "ROCM_PATH environment variable is set to ${ROCM_PATH} but does not exist.\n"
      "Set a valid ROCM_PATH or unset ROCM_PATH environment variable to fix.")
  endif()
else()
  if(UNIX)
    set(ROCM_PATH /opt/rocm)
  else() # Win32
    set(ROCM_PATH C:/opt/rocm)
  endif()
  if(NOT EXISTS ${ROCM_PATH})
    message(STATUS
        "ROCM_PATH environment variable is not set and ${ROCM_PATH} does not exist.\n"
        "Building without ROCm support.")
    return()
  endif()
endif()

# MAGMA_HOME
if(NOT DEFINED ENV{MAGMA_HOME})
  set(MAGMA_HOME ${ROCM_PATH}/magma)
  set(ENV{MAGMA_HOME} ${ROCM_PATH}/magma)
else()
  file(TO_CMAKE_PATH "$ENV{MAGMA_HOME}" MAGMA_HOME)
endif()

# MIOpen isn't a part of HIP-SDK for Windows and hence, may have a different
# installation directory.
if(WIN32)
  if(NOT DEFINED ENV{MIOPEN_PATH})
    set(miopen_DIR C:/opt/miopen/lib/cmake/miopen)
  else()
    set(miopen_DIR $ENV{MIOPEN_PATH}/lib/cmake/miopen)
  endif()
endif()

torch_hip_get_arch_list(PYTORCH_ROCM_ARCH)
if(PYTORCH_ROCM_ARCH STREQUAL "")
  message(FATAL_ERROR "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH environment variable to specify GPU archs to build for.")
endif()
message("Building PyTorch for GPU arch: ${PYTORCH_ROCM_ARCH}")

# Add HIP to the CMAKE Module Path
# needed because the find_package call to this module uses the Module mode search
# https://cmake.org/cmake/help/latest/command/find_package.html#search-modes
if(UNIX)
  set(CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${CMAKE_MODULE_PATH})
else() # Win32
  set(CMAKE_MODULE_PATH ${ROCM_PATH}/cmake/ ${CMAKE_MODULE_PATH})
endif()

# Add ROCM_PATH to CMAKE_PREFIX_PATH, needed because the find_package
# call to individual ROCM components uses the Config mode search
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  if(NOT ${PACKAGE_NAME}_FOUND)
    message("Optional package ${PACKAGE_NAME} not found")
  else()
    message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
    if(${PACKAGE_NAME}_INCLUDE_DIR)
      list(APPEND ROCM_INCLUDE_DIRS ${${PACKAGE_NAME}_INCLUDE_DIR})
    endif()
  endif()
endmacro()

# Find the HIP Package
# MODULE argument is added for clarity that CMake is searching
# for FindHIP.cmake in Module mode
find_package_and_print_version(HIP 1.0 MODULE)

if(HIP_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)
  find_package_and_print_version(hip REQUIRED CONFIG)
  if(HIP_VERSION)
    # Check if HIP_VERSION contains a dash (e.g., "7.1.25421-32f9fa6ca5")
    # and strip everything after it to get clean numeric version
    string(FIND "${HIP_VERSION}" "-" DASH_POS)
    if(NOT DASH_POS EQUAL -1)
      string(SUBSTRING "${HIP_VERSION}" 0 ${DASH_POS} HIP_VERSION_CLEAN)
      set(HIP_VERSION "${HIP_VERSION_CLEAN}")
    else()
      set(HIP_VERSION_CLEAN "${HIP_VERSION}")
    endif()
    message("HIP version: ${HIP_VERSION}")
  else()
    set(HIP_VERSION_CLEAN "")
  endif()

# The rocm-core package was only introduced in ROCm 6.4, so we make it optional.
  find_package(rocm-core CONFIG)

  # Some old consumer HIP SDKs do not distribute rocm_version.h, so we allow
  # falling back to the hip version, which everyone should have.
  # rocm_version.h lives in the rocm-core package and hip_version.h lives in the
  # hip (lower-case) package. Both are probed above and will be in
  # ROCM_INCLUDE_DIRS if available.
  find_file(ROCM_VERSION_HEADER_PATH
    NAMES rocm-core/rocm_version.h hip/hip_version.h
    NO_DEFAULT_PATH
    PATHS ${ROCM_INCLUDE_DIRS}
  )
  if(ROCM_VERSION_HEADER_PATH MATCHES "rocm-core/rocm_version.h$")
    set(ROCM_LIB_NAME "ROCM")
  else()
    set(ROCM_LIB_NAME "HIP")
  endif()

  if(NOT ROCM_VERSION_HEADER_PATH)
    message(FATAL_ERROR "Could not find hip/hip_version.h or rocm-core/rocm_version.h in ${ROCM_INCLUDE_DIRS}")
  endif()
  get_filename_component(ROCM_HEADER_NAME ${ROCM_VERSION_HEADER_PATH} NAME)

  if(EXISTS ${ROCM_VERSION_HEADER_PATH})
    set(ROCM_HEADER_FILE ${ROCM_VERSION_HEADER_PATH})
  else()
    message(FATAL_ERROR "********************* ${ROCM_HEADER_NAME} could not be found ******************\n")
  endif()

  # Read the ROCM headerfile into a variable
  message(STATUS "Reading ROCM version from: ${ROCM_HEADER_FILE}")
  message(STATUS "Content: ${ROCM_HEADER_CONTENT}")
  file(READ "${ROCM_HEADER_FILE}" ROCM_HEADER_CONTENT)

  # Below we use a RegEx to find ROCM version numbers.
  # Note that CMake does not support \s for blank space. That is
  # why in the regular expressions below we have a blank space in
  # the square brackets.
  # There are three steps:
  # 1. Match regular expression
  # 2. Strip the non-numerical part of the string
  # 3. Strip leading and trailing spaces

  string(REGEX MATCH "${ROCM_LIB_NAME}_VERSION_MAJOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "${ROCM_LIB_NAME}_VERSION_MAJOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MAJOR)
  string(REGEX MATCH "${ROCM_LIB_NAME}_VERSION_MINOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "${ROCM_LIB_NAME}_VERSION_MINOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MINOR)
  string(REGEX MATCH "${ROCM_LIB_NAME}_VERSION_PATCH[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "${ROCM_LIB_NAME}_VERSION_PATCH" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_PATCH)

  # Create ROCM_VERSION_DEV_INT which is later used as a preprocessor macros
  set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
  math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")

  message("\n***** ROCm version from ${ROCM_HEADER_NAME} ****\n")
  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message("ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")

  math(EXPR TORCH_HIP_VERSION "(${HIP_VERSION_MAJOR} * 100) + ${HIP_VERSION_MINOR}")
  message("HIP_VERSION_MAJOR: ${HIP_VERSION_MAJOR}")
  message("HIP_VERSION_MINOR: ${HIP_VERSION_MINOR}")
  message("TORCH_HIP_VERSION: ${TORCH_HIP_VERSION}")

  # Find ROCM components using Config mode
  # These components will be searced for recursively in ${ROCM_PATH}
  message("\n***** Library versions from cmake find_package *****\n")
  find_package_and_print_version(amd_comgr REQUIRED)
  find_package_and_print_version(rocrand REQUIRED)
  find_package_and_print_version(hiprand REQUIRED)
  find_package_and_print_version(rocblas REQUIRED)
  find_package_and_print_version(hipblas REQUIRED)
  find_package_and_print_version(miopen REQUIRED)
  find_package_and_print_version(hipfft REQUIRED)
  find_package_and_print_version(hipsparse REQUIRED)
  find_package_and_print_version(rocprim REQUIRED)
  find_package_and_print_version(hipcub REQUIRED)
  find_package_and_print_version(rocthrust REQUIRED)
  find_package_and_print_version(hipsolver REQUIRED)
  find_package_and_print_version(rocsolver REQUIRED)
  # workaround cmake 4 build issue
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    message(WARNING "Work around hiprtc cmake failure for cmake >= 4")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    find_package_and_print_version(hiprtc REQUIRED)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
  else()
    find_package_and_print_version(hiprtc REQUIRED)
  endif()
  find_package_and_print_version(hipblaslt REQUIRED)

  if(UNIX)
    find_package_and_print_version(rccl)
    find_package_and_print_version(hsa-runtime64 REQUIRED)
  endif()

  # Optional components.
  find_package_and_print_version(hipsparselt)  # Will be required when ready.

  list(REMOVE_DUPLICATES ROCM_INCLUDE_DIRS)

  if(UNIX)
    # roctx is part of roctracer
    find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCM_PATH}/lib)

    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")

    if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "5.7.0")
      # check whether hipblaslt provides HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
      set(file "${PROJECT_BINARY_DIR}/hipblaslt_test_outer_vec.cc")
      file(WRITE ${file} ""
        "#define LEGACY_HIPBLAS_DIRECT\n"
        "#include <hipblaslt/hipblaslt.h>\n"
        "int main() {\n"
        "    hipblasLtMatmulMatrixScale_t attr = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;\n"
        "    return 0;\n"
        "}\n"
        )
      try_compile(hipblaslt_compile_result_outer_vec ${PROJECT_RANDOM_BINARY_DIR} ${file}
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ROCM_INCLUDE_DIRS}"
        COMPILE_DEFINITIONS -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
        OUTPUT_VARIABLE hipblaslt_compile_output_outer_vec)

      # check whether hipblaslt provides HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT
      set(file "${PROJECT_BINARY_DIR}/hipblaslt_test_vec_ext.cc")
      file(WRITE ${file} ""
        "#define LEGACY_HIPBLAS_DIRECT\n"
        "#include <hipblaslt/hipblaslt.h>\n"
        "int main() {\n"
        "    hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT;\n"
        "    return 0;\n"
        "}\n"
        )
      try_compile(hipblaslt_compile_result_vec_ext ${PROJECT_RANDOM_BINARY_DIR} ${file}
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ROCM_INCLUDE_DIRS}"
        COMPILE_DEFINITIONS -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
        OUTPUT_VARIABLE hipblaslt_compile_output_vec_ext)

      if(hipblaslt_compile_result_outer_vec)
        set(HIPBLASLT_OUTER_VEC ON)
        set(HIPBLASLT_VEC_EXT OFF)
        message("hipblaslt is using scale pointer outer vec")
      elseif(hipblaslt_compile_result_vec_ext)
        set(HIPBLASLT_OUTER_VEC OFF)
        set(HIPBLASLT_VEC_EXT ON)
        message("hipblaslt is using scale pointer vec ext")
      else()
        set(HIPBLASLT_OUTER_VEC OFF)
        set(HIPBLASLT_VEC_EXT OFF)
        message("hipblaslt is NOT using scale pointer outer vec: ${hipblaslt_compile_output_outer_vec}")
        message("hipblaslt is NOT using scale pointer vec ext: ${hipblaslt_compile_output_vec_ext}")
      endif()
    endif()
  endif()
endif()
