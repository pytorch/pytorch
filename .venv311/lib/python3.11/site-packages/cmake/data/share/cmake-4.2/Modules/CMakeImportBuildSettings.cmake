# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is purposely no longer documented.  It does nothing useful.

# This macro used to load build settings from another project that
# stored settings using the CMAKE_EXPORT_BUILD_SETTINGS macro.
macro(CMAKE_IMPORT_BUILD_SETTINGS SETTINGS_FILE)
  if("${SETTINGS_FILE}" STREQUAL "")
    message(SEND_ERROR "CMAKE_IMPORT_BUILD_SETTINGS called with no argument.")
  endif()
endmacro()
