# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

include(Platform/Linker/GNU)

macro(__linux_linker_gnu lang)
  set(CMAKE_${lang}_PLATFORM_LINKER_ID GNU)
  if(CMAKE_${lang}_COMPILER_LINKER)
    __cmake_set_whole_archive_feature("${CMAKE_${lang}_COMPILER_LINKER}" "${lang}")
  endif()
endmacro()
