# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
include_guard()

macro(__aix_linker_aix lang)
  set(CMAKE_${lang}_PLATFORM_LINKER_ID AIX)
  set(CMAKE_${lang}_LINK_LIBRARIES_PROCESSING ORDER=REVERSE DEDUPLICATION=ALL)
endmacro()
