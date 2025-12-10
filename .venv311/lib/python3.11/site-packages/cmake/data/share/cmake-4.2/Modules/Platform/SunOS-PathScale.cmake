# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__SUNOS_COMPILER_PATHSCALE)
  return()
endif()
set(__SUNOS_COMPILER_PATHSCALE 1)

macro(__sunos_compiler_pathscale lang)
  # Shared library compile and link flags.
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")

  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,-R")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,-h")
endmacro()
