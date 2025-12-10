# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_Diab)
  return()
endif()
set(__COMPILER_Diab 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_diab lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-#")
  set(CMAKE_${lang}_OUTPUT_EXTENSION ".o")

  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O -Xsize-opt")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -XO")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -XO -g3")

  set(__DIAB_AR "${CMAKE_${lang}_COMPILER_AR}")
  set(CMAKE_${lang}_CREATE_STATIC_LIBRARY "\"${__DIAB_AR}\"  -r <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_CREATE "\"${__DIAB_AR}\" -r <TARGET> <LINK_FLAGS> <OBJECTS>")

  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
  set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
  set(CMAKE_${lang}_COMPILE_OPTIONS_IPO -XO -Xwhole-program-optim)
endmacro()

set(CMAKE_EXECUTABLE_SUFFIX "")
set(CMAKE_LIBRARY_PATH_TERMINATOR "")
set(CMAKE_LIBRARY_PATH_FLAG "")
