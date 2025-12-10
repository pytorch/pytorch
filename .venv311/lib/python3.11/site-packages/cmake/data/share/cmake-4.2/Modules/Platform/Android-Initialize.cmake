# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# When CMAKE_SYSTEM_NAME is "Android", CMakeSystemSpecificInitialize loads this
# module.

# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/pre/Android-Initialize.cmake OPTIONAL)
endif()

include(Platform/Linux-Initialize)
unset(LINUX)

set(ANDROID 1)

# Support for NVIDIA Nsight Tegra Visual Studio Edition was previously
# implemented in the CMake VS IDE generators.  Avoid interfering with
# that functionality for now.
if(CMAKE_VS_PLATFORM_NAME STREQUAL "Tegra-Android")
  return()
endif()

# Commonly used Android toolchain files that pre-date CMake upstream support
# set CMAKE_SYSTEM_VERSION to 1.  Avoid interfering with them.
if(CMAKE_SYSTEM_VERSION EQUAL 1)
  return()
endif()

set(CMAKE_BUILD_TYPE_INIT "RelWithDebInfo")

if(CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED)
  # Tell CMake not to search host sysroots for headers/libraries.

  # All paths added to CMAKE_SYSTEM_*_PATH below will be rerooted under
  # CMAKE_FIND_ROOT_PATH. This is set because:
  # 1. Users may structure their libraries in a way similar to NDK. When they do that,
  #    they can simply append another path to CMAKE_FIND_ROOT_PATH.
  # 2. CMAKE_FIND_ROOT_PATH must be non-empty for CMAKE_FIND_ROOT_PATH_MODE_* == ONLY
  #    to be meaningful. https://github.com/android-ndk/ndk/issues/890
  list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/sysroot")

  # Allow users to override these values in case they want more strict behaviors.
  # For example, they may want to prevent the NDK's libz from being picked up so
  # they can use their own.
  # https://github.com/android-ndk/ndk/issues/517
  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  endif()

  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  endif()

  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  endif()

  if(NOT DEFINED CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
  endif()

  # Allows CMake to find headers in the architecture-specific include directories.
  set(CMAKE_LIBRARY_ARCHITECTURE "${CMAKE_ANDROID_ARCH_TRIPLE}")

  # Instructs CMake to search the correct API level for libraries.
  # Besides the paths like <root>/<prefix>/lib/<arch>, cmake also searches <root>/<prefix>.
  # So we can add the API level specific directory directly.
  # https://github.com/android/ndk/issues/929
  list(PREPEND CMAKE_SYSTEM_PREFIX_PATH
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/${CMAKE_SYSTEM_VERSION}"
    )

  list(APPEND CMAKE_SYSTEM_PROGRAM_PATH "${CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED}/bin")
endif()

# Skip sysroot selection if the NDK has a unified toolchain.
if(CMAKE_ANDROID_NDK_TOOLCHAIN_UNIFIED)
  return()
endif()

# Natively compiling on an Android host doesn't use the NDK cross-compilation
# tools.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Android")
  return()
endif()

if(NOT CMAKE_SYSROOT)
  if(CMAKE_ANDROID_NDK)
    set(CMAKE_SYSROOT "${CMAKE_ANDROID_NDK}/platforms/android-${CMAKE_SYSTEM_VERSION}/arch-${CMAKE_ANDROID_ARCH}")
    if(NOT CMAKE_ANDROID_NDK_DEPRECATED_HEADERS)
      set(CMAKE_SYSROOT_COMPILE "${CMAKE_ANDROID_NDK}/sysroot")
    endif()
  elseif(CMAKE_ANDROID_STANDALONE_TOOLCHAIN)
    set(CMAKE_SYSROOT "${CMAKE_ANDROID_STANDALONE_TOOLCHAIN}/sysroot")
  endif()
endif()

if(CMAKE_SYSROOT)
  if(NOT IS_DIRECTORY "${CMAKE_SYSROOT}")
    message(FATAL_ERROR
      "Android: The system root directory needed for the selected Android version and architecture does not exist:\n"
      "  ${CMAKE_SYSROOT}\n"
      )
  endif()
else()
  message(FATAL_ERROR
    "Android: No CMAKE_SYSROOT was selected."
    )
endif()

# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/post/Android-Initialize.cmake OPTIONAL)
endif()
