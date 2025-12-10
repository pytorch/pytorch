# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_PATHSCALE)
  return()
endif()
set(__COMPILER_PATHSCALE 1)

macro(__compiler_pathscale lang)
  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")

  set(CMAKE_${lang}_LINK_MODE DRIVER)

  # Initial configuration flags.
  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g -O0")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -Os")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -g -O2")
endmacro()
