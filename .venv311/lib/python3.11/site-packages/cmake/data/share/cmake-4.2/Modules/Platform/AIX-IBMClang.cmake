# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__AIX_COMPILER_IBMCLANG)
  return()
endif()
set(__AIX_COMPILER_IBMCLANG 1)

include(Platform/AIX-GNU)

macro(__aix_compiler_ibmclang lang)
  __aix_compiler_gnu(${lang})
  unset(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY)
endmacro()
