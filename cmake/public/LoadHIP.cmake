set(PYTORCH_FOUND_HIP FALSE)

# If ROCM_PATH is set, assume intention is to compile with
# ROCm support and error out if the ROCM_PATH does not exist.
# Else ROCM_PATH does not exist, assume a default of /opt/rocm
# In the latter case, if /opt/rocm does not exist emit status
# message and return.
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
  if(NOT EXISTS ${ROCM_PATH})
    message(FATAL_ERROR
      "ROCM_PATH environment variable is set to ${ROCM_PATH} but does not exist.\n"
      "Set a valid ROCM_PATH or unset ROCM_PATH environment variable to fix.")
  endif()
else()
  set(ROCM_PATH /opt/rocm)
  if(NOT EXISTS ${ROCM_PATH})
    message(STATUS
        "ROCM_PATH environment variable is not set and ${ROCM_PATH} does not exist.\n"
        "Building without ROCm support.")
    return()
  endif()
endif()

if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
  set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
else()
  set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
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
# needed because the find_package call to this module uses the Module mode search
# https://cmake.org/cmake/help/latest/command/find_package.html#search-modes
set(CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${CMAKE_MODULE_PATH})

# Add ROCM_PATH to CMAKE_PREFIX_PATH, needed because the find_package
# call to individual ROCM components uses the Config mode search
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
endmacro()

# Find the HIP Package
# MODULE argument is added for clarity that CMake is searching
# for FindHIP.cmake in Module mode
find_package_and_print_version(HIP 1.0 MODULE)

if(HIP_FOUND)
  set(PYTORCH_FOUND_HIP TRUE)

  # Find ROCM version for checks
  if(EXISTS ${ROCM_INCLUDE_DIRS}/rocm-core/rocm_version.h)
    set(ROCM_HEADER_FILE ${ROCM_INCLUDE_DIRS}/rocm-core/rocm_version.h)
  else()
    message(FATAL_ERROR "********************* rocm_version.h could not be found ******************\n")
  endif()

  # Read the ROCM headerfile into a variable
  file(READ ${ROCM_HEADER_FILE} ROCM_HEADER_CONTENT)

  # Below we use a RegEx to find ROCM version numbers.
  # Note that CMake does not support \s for blank space. That is
  # why in the regular expressions below we have a blank space in
  # the square brackets.
  # There are three steps:
  # 1. Match regular expression
  # 2. Strip the non-numerical part of the string
  # 3. Strip leading and trailing spaces
  string(REGEX MATCH "ROCM_VERSION_MAJOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_MAJOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MAJOR)
  string(REGEX MATCH "ROCM_VERSION_MINOR[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_MINOR" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_MINOR)
  string(REGEX MATCH "ROCM_VERSION_PATCH[ ]+[0-9]+" TEMP1 ${ROCM_HEADER_CONTENT})
  string(REPLACE "ROCM_VERSION_PATCH" "" TEMP2 ${TEMP1})
  string(STRIP ${TEMP2} ROCM_VERSION_DEV_PATCH)

  # Create ROCM_VERSION_DEV_INT which is later used as a preprocessor macros
  set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
  math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")

  message("\n***** ROCm version from rocm_version.h ****\n")
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

  # roctx is part of roctracer
  find_library(ROCM_ROCTX_LIB roctx64 HINTS ${ROCM_PATH}/lib)

  # check whether HIP declares new types
  set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
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
