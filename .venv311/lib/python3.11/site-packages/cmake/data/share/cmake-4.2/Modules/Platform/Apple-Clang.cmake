# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__apple_compiler_clang lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-v -Wl,-v") # also tell linker to print verbose output
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-dynamiclib -Wl,-headerpad_max_install_names")
  set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "-bundle -Wl,-headerpad_max_install_names")
  set(CMAKE_${lang}_SYSROOT_FLAG "-isysroot")
  set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mmacosx-version-min=")
  if(NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 3.2)
    set(CMAKE_${lang}_SYSTEM_FRAMEWORK_SEARCH_FLAG "-iframework ")
  endif()

  set(CMAKE_${lang}_LINK_LIBRARY_USING_FRAMEWORK "-framework <LIBRARY>")
  set(CMAKE_${lang}_LINK_LIBRARY_USING_FRAMEWORK_SUPPORTED TRUE)
  set(CMAKE_${lang}_LINK_LIBRARY_FRAMEWORK_ATTRIBUTES LIBRARY_TYPE=STATIC,SHARED DEDUPLICATION=DEFAULT OVERRIDE=DEFAULT)

  # linker selection
  set(CMAKE_${lang}_USING_LINKER_SYSTEM "-fuse-ld=ld")
  set(CMAKE_${lang}_USING_LINKER_APPLE_CLASSIC "-fuse-ld=ld" "LINKER:-ld_classic")
  set(CMAKE_${lang}_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_${lang}_USING_LINKER_MOLD "-fuse-ld=mold")
  set(CMAKE_${lang}_USING_LINKER_SOLD "-fuse-ld=sold")

  if(NOT CMAKE_${lang}_COMPILER_APPLE_SYSROOT)
    set(CMAKE_${lang}_COMPILER_APPLE_SYSROOT_REQUIRED 1)
  endif()

  if(_CMAKE_OSX_SYSROOT_PATH MATCHES "/iPhoneOS")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-miphoneos-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/iPhoneSimulator")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mios-simulator-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/AppleTVOS")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mtvos-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/AppleTVSimulator")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mtvos-simulator-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/XROS")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "--target=<ARCH>-apple-xros<VERSION_MIN>")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/XRSimulator")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "--target=<ARCH>-apple-xros<VERSION_MIN>-simulator")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/WatchOS")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mwatchos-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/WatchSimulator")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mwatchos-simulator-version-min=")
  elseif(_CMAKE_OSX_SYSROOT_PATH MATCHES "/MacOSX" AND CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "--target=<ARCH>-apple-ios<VERSION_MIN>-macabi")
  else()
    set(CMAKE_${lang}_OSX_DEPLOYMENT_TARGET_FLAG "-mmacosx-version-min=")
  endif()
endmacro()
