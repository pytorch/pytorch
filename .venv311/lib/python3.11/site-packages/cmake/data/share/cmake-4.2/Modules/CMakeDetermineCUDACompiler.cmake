# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)
include(${CMAKE_ROOT}/Modules/CMakeParseImplicitLinkInfo.cmake)

if( NOT ( ("${CMAKE_GENERATOR}" MATCHES "Make") OR
          ("${CMAKE_GENERATOR}" MATCHES "Ninja") OR
          ("${CMAKE_GENERATOR}" MATCHES "FASTBuild") OR
          ("${CMAKE_GENERATOR}" MATCHES "Visual Studio (1|[9][0-9])") ) )
  message(FATAL_ERROR "CUDA language not currently supported by \"${CMAKE_GENERATOR}\" generator")
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  if(DEFINED ENV{CUDAHOSTCXX} OR DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "Visual Studio does not support specifying CUDAHOSTCXX or CMAKE_CUDA_HOST_COMPILER. Using the C++ compiler provided by Visual Studio.")
  endif()
else()
  if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER_INIT NOTFOUND)

    # prefer the environment variable CUDACXX
    if(NOT $ENV{CUDACXX} STREQUAL "")
      get_filename_component(CMAKE_CUDA_COMPILER_INIT $ENV{CUDACXX} PROGRAM PROGRAM_ARGS CMAKE_CUDA_FLAGS_ENV_INIT)
      if(CMAKE_CUDA_FLAGS_ENV_INIT)
        set(CMAKE_CUDA_COMPILER_ARG1 "${CMAKE_CUDA_FLAGS_ENV_INIT}" CACHE STRING "Arguments to CUDA compiler")
      endif()
      if(NOT EXISTS ${CMAKE_CUDA_COMPILER_INIT})
        message(FATAL_ERROR "Could not find compiler set in environment variable CUDACXX:\n$ENV{CUDACXX}.\n${CMAKE_CUDA_COMPILER_INIT}")
      endif()
    endif()

    # finally list compilers to try
    if(NOT CMAKE_CUDA_COMPILER_INIT)
      set(CMAKE_CUDA_COMPILER_LIST nvcc)
    endif()

    set(_CMAKE_CUDA_COMPILER_PATHS "$ENV{CUDA_PATH}/bin")
    _cmake_find_compiler(CUDA)
    unset(_CMAKE_CUDA_COMPILER_PATHS)
  else()
    _cmake_find_compiler_path(CUDA)
  endif()

  mark_as_advanced(CMAKE_CUDA_COMPILER)

  #Allow the user to specify a host compiler except for Visual Studio
  if(NOT $ENV{CUDAHOSTCXX} STREQUAL "")
    get_filename_component(CMAKE_CUDA_HOST_COMPILER $ENV{CUDAHOSTCXX} PROGRAM)
    if(NOT EXISTS ${CMAKE_CUDA_HOST_COMPILER})
      message(FATAL_ERROR "Could not find compiler set in environment variable CUDAHOSTCXX:\n$ENV{CUDAHOSTCXX}.\n${CMAKE_CUDA_HOST_COMPILER}")
    endif()
  elseif(CMAKE_CUDA_HOST_COMPILER)
    # We get here if CMAKE_CUDA_HOST_COMPILER was specified by the user or toolchain file.
    if(IS_ABSOLUTE "${CMAKE_CUDA_HOST_COMPILER}")
      # Convert to forward slashes.
      cmake_path(CONVERT "${CMAKE_CUDA_HOST_COMPILER}" TO_CMAKE_PATH_LIST CMAKE_CUDA_HOST_COMPILER NORMALIZE)
    else()
      # Convert to absolute path so changes in `PATH` do not impact CUDA compilation.
      find_program(_CMAKE_CUDA_HOST_COMPILER_PATH NO_CACHE NAMES "${CMAKE_CUDA_HOST_COMPILER}")
      if(_CMAKE_CUDA_HOST_COMPILER_PATH)
        set(CMAKE_CUDA_HOST_COMPILER "${_CMAKE_CUDA_HOST_COMPILER_PATH}")
      endif()
      unset(_CMAKE_CUDA_HOST_COMPILER_PATH)
    endif()
    if(NOT EXISTS "${CMAKE_CUDA_HOST_COMPILER}")
      message(FATAL_ERROR "Could not find compiler set in variable CMAKE_CUDA_HOST_COMPILER:\n  ${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    # If the value was cached, update the cache entry with our modifications.
    get_property(_CMAKE_CUDA_HOST_COMPILER_CACHED CACHE CMAKE_CUDA_HOST_COMPILER PROPERTY TYPE)
    if(_CMAKE_CUDA_HOST_COMPILER_CACHED)
      set_property(CACHE CMAKE_CUDA_HOST_COMPILER PROPERTY VALUE "${CMAKE_CUDA_HOST_COMPILER}")
      mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)
    endif()
    unset(_CMAKE_CUDA_HOST_COMPILER_CACHED)
  endif()
endif()

if(NOT "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures")
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_CUDA_COMPILER_ID_RUN)
  set(CMAKE_CUDA_COMPILER_ID_RUN 1)

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)

  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # We will not know CMAKE_CUDA_COMPILER until the main compiler id step
    # below extracts it, but we do know that the compiler id will be NVIDIA.
    set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
  else()
    # We determine the vendor to help with find the toolkit and use the right flags for detection right away.
    # The main compiler identification is still needed below to extract other information.
    list(APPEND CMAKE_CUDA_COMPILER_ID_VENDORS NVIDIA Clang)
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_NVIDIA "nvcc: [^\n]+ Cuda compiler driver")
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_Clang "(clang version)")
    CMAKE_DETERMINE_COMPILER_ID_VENDOR(CUDA "--version")

    # Find the CUDA toolkit to get:
    # - CMAKE_CUDA_COMPILER_TOOLKIT_VERSION
    # - CMAKE_CUDA_COMPILER_TOOLKIT_ROOT
    # - CMAKE_CUDA_COMPILER_LIBRARY_ROOT
    # We save them in CMakeCUDACompiler.cmake so FindCUDAToolkit can
    # avoid searching on future runs and the toolkit is the same.
    # Match arguments with cmake_cuda_architectures_all call.
    include(Internal/CMakeCUDAFindToolkit)
    cmake_cuda_find_toolkit(CUDA CMAKE_CUDA_COMPILER_)

    set(CMAKE_CUDA_DEVICE_LINKER "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin/nvlink${CMAKE_EXECUTABLE_SUFFIX}")
    set(CMAKE_CUDA_FATBINARY "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin/fatbinary${CMAKE_EXECUTABLE_SUFFIX}")
  endif()

  set(CMAKE_CUDA_COMPILER_ID_FLAGS_ALWAYS "-v")

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(nvcc_test_flags "--keep --keep-dir tmp")
    if(CMAKE_CUDA_HOST_COMPILER)
      string(APPEND nvcc_test_flags " -ccbin=\"${CMAKE_CUDA_HOST_COMPILER}\"")
    endif()
    # If we have extracted the vendor as NVIDIA we should require detection to
    # work. If we don't, users will get confusing errors later about failure
    # to detect a default value for CMAKE_CUDA_ARCHITECTURES
    set(CMAKE_CUDA_COMPILER_ID_REQUIRE_SUCCESS ON)
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(clang_test_flags "--cuda-path=\"${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}\"")
    if(CMAKE_CROSSCOMPILING)
      # Need to pass the host target and include directories if we're crosscompiling.
      string(APPEND clang_test_flags " --sysroot=\"${CMAKE_SYSROOT}\" --target=${CMAKE_CUDA_COMPILER_TARGET}")
    endif()
  endif()

  # If the user set CMAKE_CUDA_ARCHITECTURES, validate its value.
  include(Internal/CMakeCUDAArchitecturesValidate)
  cmake_cuda_architectures_validate(CUDA)

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # Clang does not automatically select an architecture supported by the SDK.
    # Prefer NVCC's default for each SDK version, and fall back to older archs.
    set(archs "")
    if(NOT CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_LESS 13.0)
      list(APPEND archs 75)
    endif()
    if(NOT CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_LESS 11.0)
      list(APPEND archs 52)
    endif()
    list(APPEND archs 30 20)
    foreach(arch IN LISTS archs)
      list(APPEND CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "${clang_test_flags} --cuda-gpu-arch=sm_${arch}")
    endforeach()
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    list(APPEND CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "${nvcc_test_flags}")
  endif()

  # We perform compiler identification for a second time to extract implicit linking info and host compiler for NVCC.
  # We need to unset the compiler ID otherwise CMAKE_DETERMINE_COMPILER_ID() doesn't work.
  set(CMAKE_CUDA_COMPILER_ID)
  set(CMAKE_CUDA_PLATFORM_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in
    CMAKE_CUDA_COMPILER_ID_PLATFORM_CONTENT)

  CMAKE_DETERMINE_COMPILER_ID(CUDA CUDAFLAGS CMakeCUDACompilerId.cu)

  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # Now that we have the path to nvcc, we can compute the toolkit root.
    get_filename_component(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}" DIRECTORY)
    set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")

    # The compiler comes with the toolkit, so the versions are the same.
    set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
  endif()

  include(Internal/CMakeCUDAArchitecturesAll)
  # From CMAKE_CUDA_COMPILER_TOOLKIT_VERSION and CMAKE_CUDA_COMPILER_{ID,VERSION}, get:
  # - CMAKE_CUDA_ARCHITECTURES_ALL
  # - CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR
  # Match arguments with cmake_cuda_find_toolkit call.
  cmake_cuda_architectures_all(CUDA CMAKE_CUDA_COMPILER_)

  _cmake_find_compiler_sysroot(CUDA)
endif()

set(_CMAKE_PROCESSING_LANGUAGE "CUDA")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_CUDA_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(MSVC_CUDA_ARCHITECTURE_ID)
  set(SET_MSVC_CUDA_ARCHITECTURE_ID
    "set(MSVC_CUDA_ARCHITECTURE_ID ${MSVC_CUDA_ARCHITECTURE_ID})")
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  string(REGEX MATCHALL "-target-cpu sm_([0-9]+)" _clang_target_cpus "${CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT}")

  foreach(_clang_target_cpu ${_clang_target_cpus})
    if(_clang_target_cpu MATCHES "-target-cpu sm_([0-9]+)")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_DEFAULT "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  # Find target directory when crosscompiling.
  if(CMAKE_CROSSCOMPILING)
    if(CMAKE_CUDA_COMPILER_TARGET MATCHES "^([^-]+)(-|$)")
      set(_CUDA_TARGET_PROCESSOR "${CMAKE_MATCH_1}")
    elseif(CMAKE_SYSTEM_PROCESSOR)
      set(_CUDA_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
    else()
      message(FATAL_ERROR "Cross-compiling CUDA with Clang requires CMAKE_CUDA_COMPILER_TARGET and/or CMAKE_SYSTEM_PROCESSOR to be set.")
    endif()
    # Keep in sync with equivalent table in FindCUDAToolkit!
    if(_CUDA_TARGET_PROCESSOR STREQUAL "armv7-a")
      # Support for NVPACK
      set(_CUDA_TARGET_NAMES "armv7-linux-androideabi")
    elseif(_CUDA_TARGET_PROCESSOR MATCHES "arm")
      set(_CUDA_TARGET_NAMES "armv7-linux-gnueabihf")
    elseif(_CUDA_TARGET_PROCESSOR MATCHES "aarch64")
      if(ANDROID_ARCH_NAME STREQUAL "arm64")
        set(_CUDA_TARGET_NAMES "aarch64-linux-androideabi")
      elseif (CMAKE_SYSTEM_NAME STREQUAL "QNX")
        set(_CUDA_TARGET_NAMES "aarch64-qnx")
      else()
        set(_CUDA_TARGET_NAMES "aarch64-linux" "sbsa-linux")
      endif()
    elseif(_CUDA_TARGET_PROCESSOR STREQUAL "x86_64")
      set(_CUDA_TARGET_NAMES "x86_64-linux")
    endif()

    foreach(_CUDA_TARGET_NAME IN LISTS _CUDA_TARGET_NAMES)
      if(EXISTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/${_CUDA_TARGET_NAME}")
        set(_CUDA_TARGET_DIR "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/${_CUDA_TARGET_NAME}")
        break()
      endif()
    endforeach()
    unset(_CUDA_TARGET_NAME)
    unset(_CUDA_TARGET_NAMES)
    unset(_CUDA_TARGET_PROCESSOR)
  endif()

  # If not already set we can simply use the toolkit root or it's a scattered installation.
  if(NOT _CUDA_TARGET_DIR)
    set(_CUDA_TARGET_DIR "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
  endif()

  # We can't use find_library() yet at this point, so try a few guesses.
  if(EXISTS "${_CUDA_TARGET_DIR}/lib64")
    set(_CUDA_LIBRARY_DIR "${_CUDA_TARGET_DIR}/lib64")
  elseif(EXISTS "${_CUDA_TARGET_DIR}/lib/x64")
    set(_CUDA_LIBRARY_DIR "${_CUDA_TARGET_DIR}/lib/x64")
  elseif(EXISTS "${_CUDA_TARGET_DIR}/lib")
    set(_CUDA_LIBRARY_DIR "${_CUDA_TARGET_DIR}/lib")
  else()
    message(FATAL_ERROR "Unable to find _CUDA_LIBRARY_DIR based on _CUDA_TARGET_DIR=${_CUDA_TARGET_DIR}")
  endif()

  # _CUDA_TARGET_DIR always points to the directory containing the include directory.
  # On a scattered installation /usr, on a non-scattered something like /usr/local/cuda or /usr/local/cuda-10.2/targets/aarch64-linux.
  if(EXISTS "${_CUDA_TARGET_DIR}/include/cuda_runtime.h")
    set(_CUDA_INCLUDE_DIRS "${_CUDA_TARGET_DIR}/include")
  else()
    message(FATAL_ERROR "Unable to find cuda_runtime.h in \"${_CUDA_TARGET_DIR}/include\" for _CUDA_INCLUDE_DIRS.")
  endif()

  # CUDA 13 has multiple includes that are implicitly added by nvcc that we need to replicate for
  # clang-cuda
  if(EXISTS "${_CUDA_TARGET_DIR}/include/cccl")
    list(APPEND _CUDA_INCLUDE_DIRS "${_CUDA_TARGET_DIR}/include/cccl")
  endif()

  # Clang does not add any CUDA SDK libraries or directories when invoking the host linker.
  # Add the CUDA toolkit library directory ourselves so that linking works.
  # The CUDA runtime libraries are handled elsewhere by CMAKE_CUDA_RUNTIME_LIBRARY.
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${_CUDA_INCLUDE_DIRS}")
  set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "${_CUDA_LIBRARY_DIR}")
  set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
  set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

  # Don't leak variables unnecessarily to user code.
  unset(_CUDA_INCLUDE_DIRS)
  unset(_CUDA_LIBRARY_DIR)
  unset(_CUDA_TARGET_DIR)
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  include(Internal/CMakeNVCCParseImplicitInfo)
  # Parse CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT to get:
  # - CMAKE_CUDA_ARCHITECTURES_DEFAULT
  # - CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES
  # - CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
  # - CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES
  # - CMAKE_CUDA_HOST_LINK_LAUNCHER
  # - CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT
  # - CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
  # Match arguments with cmake_nvcc_filter_implicit_info call in CMakeTestCUDACompiler.
  cmake_nvcc_parse_implicit_info(CUDA CMAKE_CUDA_)

  set(_SET_CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT
    "set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT \"${CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT}\")")
endif()

include(Internal/CMakeCUDAFilterImplicitLibs)
# Filter out implicit link libraries that should not be passed unconditionally.
cmake_cuda_filter_implicit_libs(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES)

if(CMAKE_CUDA_COMPILER_SYSROOT)
  string(CONCAT _SET_CMAKE_CUDA_COMPILER_SYSROOT
    "set(CMAKE_CUDA_COMPILER_SYSROOT \"${CMAKE_CUDA_COMPILER_SYSROOT}\")\n"
    "set(CMAKE_COMPILER_SYSROOT \"${CMAKE_CUDA_COMPILER_SYSROOT}\")")
else()
  set(_SET_CMAKE_CUDA_COMPILER_SYSROOT "")
endif()

# If the user did not set CMAKE_CUDA_ARCHITECTURES, use the compiler's default.
if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
  cmake_policy(GET CMP0104 _CUDA_CMP0104)
  if(CMAKE_CUDA_COMPILER_ID AND (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR _CUDA_CMP0104 STREQUAL "NEW"))
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING "CUDA architectures")
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      message(FATAL_ERROR "Failed to detect a default CUDA architecture.\n\nCompiler output:\n${CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT}")
    endif()
  endif()
endif()
unset(CMAKE_CUDA_ARCHITECTURES_DEFAULT)

# configure all variables set in this file
configure_file(${CMAKE_ROOT}/Modules/CMakeCUDACompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCUDACompiler.cmake
  @ONLY
)

set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")
