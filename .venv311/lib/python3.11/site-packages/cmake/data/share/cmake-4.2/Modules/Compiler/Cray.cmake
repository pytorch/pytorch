# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_CRAY)
  return()
endif()
set(__COMPILER_CRAY 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_cray lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC -h PIC)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE -h PIC)
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-h PIC")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()
