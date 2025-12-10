# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckFortranSourceCompiles
--------------------------

.. versionadded:: 3.1

This module provides a command to check whether a Fortran source can be
built.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckFortranSourceCompiles)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_fortran_source_compiles

  Checks once whether the given Fortran source code can be built:

  .. code-block:: cmake

    check_fortran_source_compiles(
      <code>
      <variable>
      [FAIL_REGEX <regexes>...]
      [SRC_EXT <extension>]
    )

  This command checks once that the source supplied in ``<code>`` can be
  compiled (and linked into an executable).  The result of the check is
  stored in the internal cache variable specified by ``<variable>``.

  The arguments are:

  ``<code>``
    Fortran source code to check.  This must be an entire program, as
    written in a file containing the body block.  All symbols used in the
    source code are expected to be declared as usual in their corresponding
    headers.

  ``<variable>``
    Variable name of an internal cache variable to store the result of the
    check, with boolean true for success and boolean false for failure.

  ``FAIL_REGEX <regexes>...``
    If this option is provided with one or more regular expressions, then
    failure is determined by checking if anything in the compiler output
    matches any of the specified regular expressions.

  ``SRC_EXT <extension>``
    .. versionadded:: 3.7

    By default, the test source file used for the check will be given a
    ``.F`` file extension.  This option can be used to override this with
    ``.<extension>`` instead - ``.F90`` is a typical choice.

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

  .. include:: /module/include/CMAKE_TRY_COMPILE_TARGET_TYPE.rst

Examples
^^^^^^^^

Checking whether the Fortran compiler supports the ``pure`` procedure
attribute:

.. code-block:: cmake

  include(CheckFortranSourceCompiles)

  check_fortran_source_compiles("
    pure subroutine foo()
    end subroutine
    program test
      call foo()
    end
  " HAVE_PURE SRC_EXT "F90")

See Also
^^^^^^^^

* The :module:`CheckSourceCompiles` module for a more general command to
  check whether source can be built.
* The :module:`CheckSourceRuns` module to check whether source can be built
  and run.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceCompiles)

macro(CHECK_Fortran_SOURCE_COMPILES SOURCE VAR)
  # Pass the SRC_EXT we used by default historically.
  # A user-provided SRC_EXT argument in ARGN will override ours.
  cmake_check_source_compiles(Fortran "${SOURCE}" ${VAR} SRC_EXT "F" ${ARGN})
endmacro()
