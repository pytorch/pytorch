# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckCSourceRuns
----------------

This module provides a command to check whether a C source can be built and
run.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckCSourceRuns)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_c_source_runs

  Checks once whether the given C source code compiles and links into an
  executable that can subsequently be run:

  .. code-block:: cmake

    check_c_source_runs(<code> <variable>)

  The C source supplied in ``<code>`` must contain at least a ``main()``
  function.  The result of the check is stored in the internal cache variable
  specified by ``<variable>``.  If the code builds and runs with exit code
  ``0``, success is indicated by a boolean true value.  Failure to build or
  run is indicated by a boolean false value, such as an empty string or an
  error message.

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

The following example demonstrates how to use this module to check whether
the C source code is supported and operational at runtime.  The result of
the check is stored in the internal cache variable ``HAVE_NORETURN``.

.. code-block:: cmake

  include(CheckCSourceRuns)

  check_c_source_runs("
    #include <stdlib.h>
    #include <stdnoreturn.h>
    noreturn void f(){ exit(0); }
    int main(void) { f(); return 1; }
  " HAVE_NORETURN)

See Also
^^^^^^^^

* The :module:`CheckSourceRuns` module for a more general command syntax.
* The :module:`CheckSourceCompiles` module to check whether a source code
  can be built.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceRuns)

macro(CHECK_C_SOURCE_RUNS SOURCE VAR)
  set(_CheckSourceRuns_old_signature 1)
  cmake_check_source_runs(C "${SOURCE}" ${VAR} ${ARGN})
  unset(_CheckSourceRuns_old_signature)
endmacro()
