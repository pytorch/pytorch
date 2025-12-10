# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file contains common code blocks used by all the linker information
# files

macro(_cmake_common_linker_platform_flags lang)
  # Define configuration for LINK_WHAT_YOU_USE feature
  if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
    if(NOT DEFINED CMAKE_${lang}_LINK_WHAT_YOU_USE_FLAG)
      set(CMAKE_${lang}_LINK_WHAT_YOU_USE_FLAG "LINKER:--no-as-needed")
    endif()
    if(NOT DEFINED CMAKE_LINK_WHAT_YOU_USE_CHECK)
      set(CMAKE_LINK_WHAT_YOU_USE_CHECK ldd -u -r)
    endif()
  endif()
endmacro ()
