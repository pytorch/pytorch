# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__ANDROID_COMPILER_GNU)
  return()
endif()
set(__ANDROID_COMPILER_GNU 1)

# Support for NVIDIA Nsight Tegra Visual Studio Edition was previously
# implemented in the CMake VS IDE generators.  Avoid interfering with
# that functionality for now.  Later we may try to integrate this.
if(CMAKE_VS_PLATFORM_NAME STREQUAL "Tegra-Android")
  macro(__android_compiler_gnu lang)
  endmacro()
  return()
endif()

# Commonly used Android toolchain files that pre-date CMake upstream support
# set CMAKE_SYSTEM_VERSION to 1.  Avoid interfering with them.
if(CMAKE_SYSTEM_VERSION EQUAL 1)
  macro(__android_compiler_gnu lang)
  endmacro()
  return()
endif()

include(Platform/Android-Common)

include(Platform/Android/abi-${CMAKE_ANDROID_ARCH_ABI}-GNU)

macro(__android_compiler_gnu lang)
  __android_compiler_common(${lang})
endmacro()
