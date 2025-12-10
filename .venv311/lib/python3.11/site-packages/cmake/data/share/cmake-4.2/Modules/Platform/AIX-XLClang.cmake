# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__AIX_COMPILER_XLCLANG)
  return()
endif()
set(__AIX_COMPILER_XLCLANG 1)

include(Platform/AIX-XL)

macro(__aix_compiler_xlclang lang)
  __aix_compiler_xl(${lang})
endmacro()
