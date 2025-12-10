# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_SUNPRO)
  return()
endif()
set(__COMPILER_SUNPRO 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_sunpro lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-errwarn=%all")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()
