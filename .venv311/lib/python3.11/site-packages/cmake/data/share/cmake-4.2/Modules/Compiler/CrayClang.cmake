# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_CRAYCLANG)
  return()
endif()
set(__COMPILER_CRAYCLANG 1)

include(Compiler/Clang)

macro (__compiler_cray_clang lang)
  set(__crayclang_ver "${CMAKE_${lang}_COMPILER_VERSION}")
  set("CMAKE_${lang}_COMPILER_VERSION" "${CMAKE_${lang}_COMPILER_VERSION_INTERNAL}")
  __compiler_clang(${lang})
  set("CMAKE_${lang}_COMPILER_VERSION" "${__crayclang_ver}")
endmacro ()
