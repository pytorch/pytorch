# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

# Features for LINK_LIBRARY generator expression
if(MSVC_VERSION GREATER "1900")
  ## WHOLE_ARCHIVE: Force loading all members of an archive
  set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:/WHOLEARCHIVE:<LIBRARY>")
  set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
  set(CMAKE_LINK_LIBRARY_WHOLE_ARCHIVE_ATTRIBUTES LIBRARY_TYPE=STATIC DEDUPLICATION=YES OVERRIDE=DEFAULT)
endif()

macro(__windows_linker_msvc lang)
  set(CMAKE_${lang}_PLATFORM_LINKER_ID MSVC)
  set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=FORWARD DEDUPLICATION=ALL)

  # Features for LINK_LIBRARY generator expression
  if(DEFINED CMAKE_${lang}_COMPILER_LINKER_VERSION)
    if (CMAKE_${lang}_COMPILER_LINKER_VERSION GREATER_EQUAL "14.10")
      ## WHOLE_ARCHIVE: Force loading all members of an archive
      set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:/WHOLEARCHIVE:<LIBRARY>")
      set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
      set(CMAKE_${lang}_LINK_LIBRARY_WHOLE_ARCHIVE_ATTRIBUTES LIBRARY_TYPE=STATIC DEDUPLICATION=YES OVERRIDE=DEFAULT)
    else()
      set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED FALSE)
    endif()
  endif()
endmacro()
