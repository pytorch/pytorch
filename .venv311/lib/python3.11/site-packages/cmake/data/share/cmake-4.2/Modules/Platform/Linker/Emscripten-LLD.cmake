# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__emscripten_linker_lld lang)
  set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE "-Wl,--whole-archive" "<LINK_ITEM>" "-Wl,--no-whole-archive")
  set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
  set(CMAKE_${lang}_LINK_LIBRARY_WHOLE_ARCHIVE_ATTRIBUTES LIBRARY_TYPE=STATIC DEDUPLICATION=YES OVERRIDE=DEFAULT)

  set(CMAKE_${lang}_PLATFORM_LINKER_ID LLD)
  set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=REVERSE DEDUPLICATION=ALL)
endmacro()
