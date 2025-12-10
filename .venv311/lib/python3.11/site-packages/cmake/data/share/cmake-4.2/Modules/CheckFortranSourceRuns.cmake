# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckFortranSourceRuns
----------------------

.. versionadded:: 3.14

This module provides a command to check whether a Fortran source can be built
and run.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckFortranSourceRuns)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_fortran_source_runs

  Checks once whether the given Fortran source compiles and links into an
  executable that can subsequently be run.

  .. code-block:: cmake

    check_fortran_source_runs(<code> <variable> [SRC_EXT <extension>])

  The Fortran source supplied in ``<code>`` must contain a Fortran ``program``
  unit.  The result of the check is stored in the internal cache variable
  specified by ``<variable>``.  If the code builds and runs with exit code
  ``0``, success is indicated by a boolean true value.  Failure to build or
  run is indicated by a boolean false value, such as an empty string or an
  error message.

  The options are:

  ``SRC_EXT <extension>``
    By default, the internal test source file used for the check will be
    given a ``.F90`` file extension.  This option can be used to change the
    extension to ``.<extension>`` instead.

  .. rubric:: Variables Affecting the Check

  The following variables may be set before calling this command to modify
  the way the check is run:

  .. include:: /module/include/CMAKE_REQUIRED_FLAGS.rst

  .. include:: /module/include/CMAKE_REQUIRED_DEFINITIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_INCLUDES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

Examples
^^^^^^^^

The following example shows how to use this module to check whether a Fortran
source code runs and store the result of the check in an internal cache
variable ``HAVE_COARRAY``:

.. code-block:: cmake

  include(CheckFortranSourceRuns)

  check_fortran_source_runs([[
    program test
    real :: x[*]
    call co_sum(x)
    end program
  ]] HAVE_COARRAY)

See Also
^^^^^^^^

* The :module:`CheckSourceRuns` module for a more general command syntax.
* The :module:`CheckSourceCompiles` module to check whether a source code
  can be built.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceRuns)

macro(CHECK_Fortran_SOURCE_RUNS SOURCE VAR)
  # Pass the SRC_EXT we used by default historically.
  # A user-provided SRC_EXT argument in ARGN will override ours.
  cmake_check_source_runs(Fortran "${SOURCE}" ${VAR} SRC_EXT "F90" ${ARGN})
endmacro()
