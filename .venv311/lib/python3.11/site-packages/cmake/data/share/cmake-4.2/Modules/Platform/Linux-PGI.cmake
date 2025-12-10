# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__LINUX_COMPILER_PGI)
  return()
endif()
set(__LINUX_COMPILER_PGI 1)

macro(__linux_compiler_pgi lang)
  # Shared library compile and link flags.
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE "")
  set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS " ")
endmacro()
