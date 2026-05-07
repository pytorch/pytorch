set(PYTORCH_FOUND_HIP FALSE)

# If ROCM_PATH is set, assume intention is to compile with
# ROCm support and error out if the ROCM_PATH does not exist.
# Else ROCM_PATH does not exist, try to get it from rocm-sdk,
# or assume a default of /opt/rocm
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
  # Try to get ROCM_PATH from rocm-sdk if available
  find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
  if(ROCM_SDK_EXECUTABLE)
    execute_process(
      COMMAND ${ROCM_SDK_EXECUTABLE} path --root
      OUTPUT_VARIABLE ROCM_SDK_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE ROCM_SDK_RESULT
      ERROR_QUIET
    )
    if(ROCM_SDK_RESULT EQUAL 0 AND EXISTS "${ROCM_SDK_PATH}")
      set(ROCM_PATH "${ROCM_SDK_PATH}")
      message(STATUS "Found ROCm installation via rocm-sdk at: ${ROCM_PATH}")
    endif()
  endif()

  # Fall back to default paths if rocm-sdk did not work
  if(NOT DEFINED ROCM_PATH OR NOT EXISTS ${ROCM_PATH})
    if(UNIX)
      set(ROCM_PATH /opt/rocm)
    else() # Win32
      set(ROCM_PATH C:/opt/rocm)
    endif()
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

# Use CMake's native HIP language support instead of FindHIP.cmake.
# Set up the HIP compiler before calling enable_language(HIP).
if(DEFINED ENV{HIP_CLANG_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_CLANG_PATH}" _hip_clang_dir)
else()
  set(_hip_clang_dir "${ROCM_PATH}/lib/llvm/bin")
endif()
if(WIN32)
  set(CMAKE_HIP_COMPILER "${_hip_clang_dir}/clang++.exe")
else()
  set(CMAKE_HIP_COMPILER "${_hip_clang_dir}/clang++")
endif()
if(NOT EXISTS "${CMAKE_HIP_COMPILER}")
  message(FATAL_ERROR "HIP compiler not found at ${CMAKE_HIP_COMPILER}")
endif()

set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})

if(WIN32)
  # On Windows, C/CXX use clang-cl (MSVC frontend) but HIP must use clang++
  # (GNU frontend). CMake's Windows-Clang platform module enforces that all
  # compilers use the same frontend variant, and the ABI detection test
  # fails because MSVC-style linker flags (/machine:x64) are passed to clang++.
  # Skip compiler detection to avoid these issues.
  set(CMAKE_HIP_COMPILER_FORCED TRUE)
  set(CMAKE_HIP_COMPILER_WORKS TRUE)
  set(CMAKE_HIP_COMPILER_ID "Clang")
  set(CMAKE_HIP_COMPILER_FRONTEND_VARIANT "GNU")
endif()

enable_language(HIP)
message(STATUS "HIP language enabled with compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

if(WIN32)
  # After enable_language(HIP), the platform module Windows-Clang-HIP.cmake
  # sets MSVC-style compile/link rules (because C/CXX use MSVC frontend).
  # Override them all with GNU-style rules for clang++.

  # Compile: use GNU-style flags (-o, -isystem) instead of MSVC-style.
  set(CMAKE_HIP_COMPILE_OBJECT
    "<CMAKE_HIP_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -x hip -c <SOURCE>")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem ")
  set(CMAKE_HIP_DEPFILE_FORMAT gcc)
  set(CMAKE_DEPFILE_FLAGS_HIP "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")

  # Link: use GNU-style clang++ syntax with -Xlinker for MSVC flags.
  # Set the wrapper flag so CMake passes MSVC linker flags via -Xlinker.
  set(CMAKE_HIP_LINKER_WRAPPER_FLAG "-Xlinker" " ")
  set(CMAKE_HIP_LINKER_WRAPPER_FLAG_SEP)
  set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld-link")

  set(CMAKE_HIP_LINK_EXECUTABLE
    "<CMAKE_HIP_COMPILER> -nostartfiles -nostdlib -fuse-ld=lld-link <FLAGS> <CMAKE_HIP_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -Xlinker /MANIFEST:EMBED -Xlinker /implib:<TARGET_IMPLIB> -Xlinker /pdb:<TARGET_PDB> -Xlinker /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> <LINK_LIBRARIES>")
  set(CMAKE_HIP_CREATE_SHARED_LIBRARY
    "<CMAKE_HIP_COMPILER> -nostartfiles -nostdlib -fuse-ld=lld-link -shared <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> -Xlinker /MANIFEST:EMBED -Xlinker /implib:<TARGET_IMPLIB> -Xlinker /pdb:<TARGET_PDB> -Xlinker /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR> <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_HIP_CREATE_SHARED_MODULE ${CMAKE_HIP_CREATE_SHARED_LIBRARY})

  # Standard libraries for Windows linking
  set(CMAKE_HIP_STANDARD_LIBRARIES_INIT "-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames")

  # Tell CMake how to handle MSVC_RUNTIME_LIBRARY for the HIP compiler.
  set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded -fms-runtime-lib=static)
  set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL -fms-runtime-lib=dll)
  set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug -fms-runtime-lib=static_dbg)
  set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL -fms-runtime-lib=dll_dbg)

  # Disable MSVC-specific debug info flags for HIP
  set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded "")
endif()

# Remove any FindHIP.cmake module paths to prevent downstream contamination.
# FindHIP.cmake overrides global variables like CMAKE_HIP_LINK_EXECUTABLE
# (pointing to hipcc_linker_cmake_helper, a bash script that doesn't exist on Windows).
list(FILTER CMAKE_MODULE_PATH EXCLUDE REGEX ".*/cmake/hip$")

set(PYTORCH_FOUND_HIP TRUE)
find_package_and_print_version(hip REQUIRED CONFIG)

# Map lowercase hip_VERSION vars (from CONFIG mode) to uppercase HIP_VERSION
# vars that the rest of PyTorch's build expects (previously set by FindHIP MODULE).
if(hip_VERSION AND NOT HIP_VERSION)
  set(HIP_VERSION "${hip_VERSION}")
  set(HIP_VERSION_MAJOR "${hip_VERSION_MAJOR}")
  set(HIP_VERSION_MINOR "${hip_VERSION_MINOR}")
  set(HIP_VERSION_PATCH "${hip_VERSION_PATCH}")
endif()

if(PYTORCH_FOUND_HIP)
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
  find_package_and_print_version(rocshmem)
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
    find_package_and_print_version(rocm_smi REQUIRED)
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
