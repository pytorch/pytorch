# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__GNU_COMPILER_GNU)
  return()
endif()
set(__GNU_COMPILER_GNU 1)

macro(__gnu_compiler_gnu lang)
  # We pass this for historical reasons.  Projects may have
  # executables that use dlopen but do not set ENABLE_EXPORTS.
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-rdynamic")

  set(CMAKE_${lang}_VERBOSE_LINK_FLAG "-Wl,-v")

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
