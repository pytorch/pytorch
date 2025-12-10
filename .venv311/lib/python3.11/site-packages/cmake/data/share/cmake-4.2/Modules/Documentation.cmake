# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Documentation
-------------

.. deprecated:: 3.18
  This module does nothing, unless policy :policy:`CMP0106` is set to ``OLD``.

This module provides support for the VTK documentation framework.  It
relies on several tools (Doxygen, Perl, etc).
#]=======================================================================]

cmake_policy(GET CMP0106 _Documentation_policy)

if (_Documentation_policy STREQUAL "NEW")
  message(FATAL_ERROR
    "Documentation.cmake is VTK-specific code and should not be used in "
    "non-VTK projects. This logic in this module is best shipped with the "
    "project using it rather than with CMake. This is now an error according "
    "to policy CMP0106.")
else ()

if (_Documentation_policy STREQUAL "")
  # Ignore the warning if the project is detected as VTK itself.
  if (NOT CMAKE_PROJECT_NAME STREQUAL "VTK" AND
      NOT PROJECT_NAME STREQUAL "VTK")
    cmake_policy(GET_WARNING CMP0106 _Documentation_policy_warning)
    message(AUTHOR_WARNING
      "${_Documentation_policy_warning}\n"
      "Documentation.cmake is VTK-specific code and should not be used in "
      "non-VTK projects. This logic in this module is best shipped with the "
      "project using it rather than with CMake.")
  endif ()
  unset(_Documentation_policy_warning)
endif ()

#
# Build the documentation ?
#
option(BUILD_DOCUMENTATION "Build the documentation (Doxygen)." OFF)
mark_as_advanced(BUILD_DOCUMENTATION)

if (BUILD_DOCUMENTATION)

  #
  # Check for the tools
  #
  find_package(UnixCommands)
  find_package(Doxygen)
  find_package(Gnuplot)
  find_package(HTMLHelp)
  find_package(Perl)
  find_package(Wget)

  option(DOCUMENTATION_HTML_HELP
    "Build the HTML Help file (CHM)." OFF)

  option(DOCUMENTATION_HTML_TARZ
    "Build a compressed tar archive of the HTML doc." OFF)

  mark_as_advanced(
    DOCUMENTATION_HTML_HELP
    DOCUMENTATION_HTML_TARZ
    )

  #
  # The documentation process is controlled by a batch file.
  # We will probably need bash to create the custom target
  #

endif ()

endif ()

unset(_Documentation_policy)
