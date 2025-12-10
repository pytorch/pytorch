# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__LINUX_COMPILER_LCC)
  return()
endif()
set(__LINUX_COMPILER_LCC 1)

macro(__linux_compiler_lcc lang)
  # We pass this for historical reasons.  Projects may have
  # executables that use dlopen but do not set ENABLE_EXPORTS.
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-rdynamic")
endmacro()
