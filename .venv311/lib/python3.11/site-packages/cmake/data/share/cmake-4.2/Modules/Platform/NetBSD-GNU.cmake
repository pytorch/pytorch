# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard()

macro(__netbsd_compiler_gnu lang)
  # linker selection
  set(CMAKE_${lang}_USING_LINKER_SYSTEM "")
  set(CMAKE_${lang}_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_${lang}_USING_LINKER_BFD "-fuse-ld=bfd")
  set(CMAKE_${lang}_USING_LINKER_GOLD "-fuse-ld=gold")
  if(NOT CMAKE_${lang}_COMPILER_ID STREQUAL "GNU"
      OR CMAKE_${lang}_COMPILER_VERSION VERSION_GREATER_EQUAL "12.1")
    set(CMAKE_${lang}_USING_LINKER_MOLD "-fuse-ld=mold")
  endif()
endmacro()
