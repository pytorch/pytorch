# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()


include(Platform/Linker/Linux-GNU)


macro(__linux_linker_mold lang)
  __linux_linker_gnu(${lang})


  set(CMAKE_C_PLATFORM_LINKER_ID MOLD)
  set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=REVERSE DEDUPLICATION=ALL)
endmacro()
