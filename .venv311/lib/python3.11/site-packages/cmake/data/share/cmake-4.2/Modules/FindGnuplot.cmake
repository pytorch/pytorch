# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGnuplot
-----------

Finds the Gnuplot command-line graphing utility for generating two- and
three-dimensional plots (``gnuplot``):

.. code-block:: cmake

  find_package(Gnuplot [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Gnuplot_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) Gnuplot was found.

``Gnuplot_VERSION``
  .. versionadded:: 4.2

  The version of Gnuplot found.

  .. note::

    Version detection is available only for Gnuplot 4 and later.  Earlier
    versions did not provide version output.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GNUPLOT_EXECUTABLE``
  Absolute path to the ``gnuplot`` executable.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``GNUPLOT_FOUND``
  .. deprecated:: 4.2
    Use ``Gnuplot_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) Gnuplot was found.

``GNUPLOT_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Gnuplot_VERSION``.

  The version of Gnuplot found.

Examples
^^^^^^^^

Finding Gnuplot and executing it in a process:

.. code-block:: cmake

  find_package(Gnuplot)
  if(Gnuplot_FOUND)
    execute_process(COMMAND ${GNUPLOT_EXECUTABLE} --help)
  endif()
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/FindCygwin.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindMsys.cmake)

find_program(GNUPLOT_EXECUTABLE
  NAMES
  gnuplot
  pgnuplot
  wgnupl32
  PATHS
  ${CYGWIN_INSTALL_PATH}/bin
  ${MSYS_INSTALL_PATH}/usr/bin
)

if (GNUPLOT_EXECUTABLE)
    execute_process(COMMAND "${GNUPLOT_EXECUTABLE}" --version
                  OUTPUT_VARIABLE GNUPLOT_OUTPUT_VARIABLE
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "^gnuplot ([0-9\\.]+)( patchlevel )?" "\\1." Gnuplot_VERSION "${GNUPLOT_OUTPUT_VARIABLE}")
    string(REGEX REPLACE "\\.$" "" Gnuplot_VERSION "${Gnuplot_VERSION}")
    set(GNUPLOT_VERSION_STRING "${Gnuplot_VERSION}")
    unset(GNUPLOT_OUTPUT_VARIABLE)
endif()

# for compatibility
set(GNUPLOT ${GNUPLOT_EXECUTABLE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gnuplot
                                  REQUIRED_VARS GNUPLOT_EXECUTABLE
                                  VERSION_VAR Gnuplot_VERSION)

mark_as_advanced( GNUPLOT_EXECUTABLE )
