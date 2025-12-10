# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_NVHPC)
  return()
endif()
set(__COMPILER_NVHPC 1)

include(Compiler/PGI)

macro(__compiler_nvhpc lang)
  # Logic specific to NVHPC.
  set(CMAKE_INCLUDE_SYSTEM_FLAG_${lang} "-isystem ")
  set(CMAKE_${lang}_COMPILE_OPTIONS_EXTERNAL_TOOLCHAIN "--gcc-toolchain=")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-Werror")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
endmacro()
