# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/pre/Android.cmake OPTIONAL)
endif()

include(Platform/Linux)

# Natively compiling on an Android host doesn't need these flags to be reset.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Android")
  return()
endif()

# NDK organizes API level specific libraries in numbered subdirectories. To
# avoid incorrect inclusion of libraries below the targeted API level, disable
# architecture specific path suffixes by default.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB32_PATHS OFF)
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS OFF)

# Conventionally Android does not use versioned soname
# But in modern versions it is acceptable
if(NOT DEFINED CMAKE_PLATFORM_NO_VERSIONED_SONAME)
  set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)
endif()

# Android reportedly ignores RPATH, and we cannot predict the install
# location anyway.
set(CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "")

# Nsight Tegra Visual Studio Edition takes care of
# prefixing library names with '-l'.
if(CMAKE_VS_PLATFORM_NAME STREQUAL "Tegra-Android")
  set(CMAKE_LINK_LIBRARY_FLAG "")
endif()

# Commonly used Android toolchain files that pre-date CMake upstream support
# set CMAKE_SYSTEM_VERSION to 1.  Avoid interfering with them.
if(CMAKE_SYSTEM_VERSION EQUAL 1)
  # The NDK legacy toolchain file provides its version number.
  set(CMAKE_ANDROID_NDK_VERSION ${ANDROID_NDK_MAJOR}.${ANDROID_NDK_MINOR})
  return()
endif()

# Include the NDK hook.
# It can be used by NDK to inject necessary fixes for an earlier cmake.
if(CMAKE_ANDROID_NDK)
  include(${CMAKE_ANDROID_NDK}/build/cmake/hooks/post/Android.cmake OPTIONAL)
endif()
