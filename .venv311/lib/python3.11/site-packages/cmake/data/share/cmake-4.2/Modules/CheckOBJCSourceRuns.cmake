# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckOBJCSourceRuns
-------------------

.. versionadded:: 3.16

This module provides a command to check whether an Objective-C source can
be built and run.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckOBJCSourceRuns)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_objc_source_runs

  Checks once whether the given Objective-C source code compiles and links
  into an executable that can subsequently be run:

  .. code-block:: cmake

    check_objc_source_runs(<code> <variable>)

  The Objective-C source supplied in ``<code>`` must contain at least a
  ``main()`` function.  The result of the check is stored in the internal
  cache variable specified by ``<variable>``.  If the code builds and runs
  with exit code ``0``, success is indicated by a boolean true value.
  Failure to build or run is indicated by a boolean false value, such as an
  empty string or an error message.

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

In the following example, this module is used to check whether the provided
Objective-C source code builds and runs.  Result of the check is stored in
an internal cache variable ``HAVE_WORKING_CODE``.

.. code-block:: cmake

  include(CheckOBJCSourceRuns)

  check_objc_source_runs("
    #import <Foundation/Foundation.h>
    int main()
    {
      NSObject *foo;
      return 0;
    }
  " HAVE_WORKING_CODE)

See Also
^^^^^^^^

* The :module:`CheckSourceRuns` module for a more general command syntax.
* The :module:`CheckSourceCompiles` module to check whether a source code
  can be built.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceRuns)

macro(CHECK_OBJC_SOURCE_RUNS SOURCE VAR)
  set(_CheckSourceRuns_old_signature 1)
  cmake_check_source_runs(OBJC "${SOURCE}" ${VAR} ${ARGN})
  unset(_CheckSourceRuns_old_signature)
endmacro()
