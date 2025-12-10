# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()


include(Platform/Linker/SerenityOS-GNU)


macro(__serenityos_linker_lld lang)
  __serenityos_linker_gnu(${lang})

  set(CMAKE_${lang}_PLATFORM_LINKER_ID LLD)
  set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=REVERSE DEDUPLICATION=ALL)
endmacro()
