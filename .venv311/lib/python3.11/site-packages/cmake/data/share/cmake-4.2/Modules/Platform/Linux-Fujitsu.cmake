# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__LINUX_COMPILER_FUJITSU)
  return()
endif()
set(__LINUX_COMPILER_FUJITSU 1)

macro(__linux_compiler_fujitsu lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
endmacro()
