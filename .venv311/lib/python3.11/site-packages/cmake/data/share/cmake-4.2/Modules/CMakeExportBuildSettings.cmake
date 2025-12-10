# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is purposely no longer documented.  It does nothing useful.
if(NOT "${CMAKE_MINIMUM_REQUIRED_VERSION}" VERSION_LESS 2.7)
  message(FATAL_ERROR
    "The functionality of this module has been dropped as of CMake 2.8.  "
    "It was deemed harmful (confusing users by changing their compiler).  "
    "Please remove calls to the CMAKE_EXPORT_BUILD_SETTINGS macro and "
    "stop including this module.  "
    "If this project generates any files for use by external projects, "
    "remove any use of the CMakeImportBuildSettings module from them.")
endif()

# This macro used to store build settings of a project in a file to be
# loaded by another project using CMAKE_IMPORT_BUILD_SETTINGS.  Now it
# creates a file that refuses to load (with comment explaining why).
macro(CMAKE_EXPORT_BUILD_SETTINGS SETTINGS_FILE)
  if(NOT ${SETTINGS_FILE} STREQUAL "")
    configure_file(${CMAKE_ROOT}/Modules/CMakeBuildSettings.cmake.in
                   ${SETTINGS_FILE} @ONLY)
  else()
    message(SEND_ERROR "CMAKE_EXPORT_BUILD_SETTINGS called with no argument.")
  endif()
endmacro()
