# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__LINUX_COMPILER_INTEL_LLVM)
  return()
endif()
set(__LINUX_COMPILER_INTEL_LLVM 1)

macro(__linux_compiler_intel_llvm lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER YES)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE ${CMAKE_${lang}_COMPILE_OPTIONS_PIE} "-pie")
  set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "-no-pie")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")

  # We pass this for historical reasons.  Projects may have
  # executables that use dlopen but do not set ENABLE_EXPORTS.
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-rdynamic")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
endmacro()
