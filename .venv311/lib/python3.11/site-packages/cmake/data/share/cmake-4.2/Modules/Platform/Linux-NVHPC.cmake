# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__linux_compiler_nvhpc lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER YES)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE "-fPIE")
  if(CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL 25.07)
    set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "-fno-pie")
  else()
    set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "")
  endif()
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "")
endmacro()
