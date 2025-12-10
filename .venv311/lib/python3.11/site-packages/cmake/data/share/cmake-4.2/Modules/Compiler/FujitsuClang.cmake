# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_FUJITSUCLANG)
  return()
endif()
set(__COMPILER_FUJITSUCLANG 1)

include(Compiler/Clang)
