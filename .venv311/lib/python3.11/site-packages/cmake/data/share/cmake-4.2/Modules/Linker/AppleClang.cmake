# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple linkers; use include blocker.

include_guard()

macro(__linker_appleclang lang)
  # Linker warning as error
  set(CMAKE_${lang}_LINK_OPTIONS_WARNING_AS_ERROR "LINKER:-fatal_warnings")
endmacro()
