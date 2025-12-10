# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_XLCLANG)
  return()
endif()
set(__COMPILER_XLCLANG 1)

include(Compiler/XL)

macro(__compiler_xlclang lang)
  __compiler_xl(${lang})

  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-V")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror")
  set(CMAKE_${lang}_RESPONSE_FILE_FLAG "@")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "@")
endmacro()
