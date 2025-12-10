# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__windows_linker_lld lang)
  set(CMAKE_${lang}_PLATFORM_LINKER_ID LLD)
  # Features for LINK_LIBRARY generator expression
  if(CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT STREQUAL "GNU")
    include(Platform/Linker/Windows-GNU)
    __windows_linker_gnu(${lang})

    set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=FORWARD DEDUPLICATION=ALL)
  else()
    include(Platform/Linker/Windows-MSVC)
    __windows_linker_msvc(${lang})
  endif()
endmacro()
