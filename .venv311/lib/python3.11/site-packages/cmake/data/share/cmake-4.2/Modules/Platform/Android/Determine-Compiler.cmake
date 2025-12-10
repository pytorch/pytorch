# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__ANDROID_DETERMINE_COMPILER)
  return()
endif()
set(__ANDROID_DETERMINE_COMPILER 1)

# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/pre/Determine-Compiler.cmake OPTIONAL)
endif()

# Support for NVIDIA Nsight Tegra Visual Studio Edition was previously
# implemented in the CMake VS IDE generators.  Avoid interfering with
# that functionality for now.  Later we may try to integrate this.
if(CMAKE_VS_PLATFORM_NAME STREQUAL "Tegra-Android")
  macro(__android_determine_compiler lang)
  endmacro()
  return()
endif()

# Commonly used Android toolchain files that pre-date CMake upstream support
# set CMAKE_SYSTEM_VERSION to 1.  Avoid interfering with them.
if(CMAKE_SYSTEM_VERSION EQUAL 1)
  macro(__android_determine_compiler lang)
  endmacro()
  return()
endif()

# Identify the host platform.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(_ANDROID_HOST_EXT "")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(_ANDROID_HOST_EXT "")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  set(_ANDROID_HOST_EXT ".exe")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Android")
  # Natively compiling on an Android host doesn't use the NDK cross-compilation
  # tools.
  macro(__android_determine_compiler lang)
    # Do nothing
  endmacro()
  if(NOT CMAKE_CXX_COMPILER_NAMES)
    set(CMAKE_CXX_COMPILER_NAMES c++)
  endif()
  return()
else()
  message(FATAL_ERROR "Android: Builds hosted on '${CMAKE_HOST_SYSTEM_NAME}' not supported.")
endif()

if(CMAKE_ANDROID_NDK)
  include(Platform/Android/Determine-Compiler-NDK)
elseif(CMAKE_ANDROID_STANDALONE_TOOLCHAIN)
  include(Platform/Android/Determine-Compiler-Standalone)
else()
  set(_ANDROID_TOOL_NDK_TOOLCHAIN_VERSION "")
  set(_ANDROID_TOOL_C_COMPILER "")
  set(_ANDROID_TOOL_C_TOOLCHAIN_MACHINE "")
  set(_ANDROID_TOOL_C_TOOLCHAIN_VERSION "")
  set(_ANDROID_TOOL_C_COMPILER_EXTERNAL_TOOLCHAIN "")
  set(_ANDROID_TOOL_C_TOOLCHAIN_PREFIX "")
  set(_ANDROID_TOOL_C_TOOLCHAIN_SUFFIX "")
  set(_ANDROID_TOOL_CXX_COMPILER "")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_MACHINE "")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_VERSION "")
  set(_ANDROID_TOOL_CXX_COMPILER_EXTERNAL_TOOLCHAIN "")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_PREFIX "")
  set(_ANDROID_TOOL_CXX_TOOLCHAIN_SUFFIX "")
endif()

unset(_ANDROID_HOST_EXT)

macro(__android_determine_compiler lang)
  if(_ANDROID_TOOL_${lang}_COMPILER)
    set(CMAKE_${lang}_COMPILER "${_ANDROID_TOOL_${lang}_COMPILER}")
    set(CMAKE_${lang}_COMPILER_EXTERNAL_TOOLCHAIN "${_ANDROID_TOOL_${lang}_COMPILER_EXTERNAL_TOOLCHAIN}")

    # Save the Android-specific information in CMake${lang}Compiler.cmake.
    set(CMAKE_${lang}_COMPILER_CUSTOM_CODE "
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION \"${_ANDROID_TOOL_NDK_TOOLCHAIN_VERSION}\")
set(CMAKE_${lang}_ANDROID_TOOLCHAIN_MACHINE \"${_ANDROID_TOOL_${lang}_TOOLCHAIN_MACHINE}\")
set(CMAKE_${lang}_ANDROID_TOOLCHAIN_VERSION \"${_ANDROID_TOOL_${lang}_TOOLCHAIN_VERSION}\")
set(CMAKE_${lang}_COMPILER_EXTERNAL_TOOLCHAIN \"${_ANDROID_TOOL_${lang}_COMPILER_EXTERNAL_TOOLCHAIN}\")
set(CMAKE_${lang}_ANDROID_TOOLCHAIN_PREFIX \"${_ANDROID_TOOL_${lang}_TOOLCHAIN_PREFIX}\")
set(CMAKE_${lang}_ANDROID_TOOLCHAIN_SUFFIX \"${_ANDROID_TOOL_${lang}_TOOLCHAIN_SUFFIX}\")
")
  endif()
endmacro()

# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/post/Determine-Compiler.cmake OPTIONAL)
endif()
