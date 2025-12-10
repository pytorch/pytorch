# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckCXXSourceCompiles
----------------------

This module provides a command to check whether a C++ source can be built.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckCXXSourceCompiles)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_cxx_source_compiles

  Checks once whether the given C++ source code can be built:

  .. code-block:: cmake

    check_cxx_source_compiles(<code> <variable> [FAIL_REGEX <regexes>...])

  This command checks once that the source supplied in ``<code>`` can be
  compiled (and linked into an executable).  The result of the check is
  stored in the internal cache variable specified by ``<variable>``.

  The arguments are:

  ``<code>``
    C++ source code to check.  This must be an entire program, as written
    in a file containing the body block.  All symbols used in the source code
    are expected to be declared as usual in their corresponding headers.

  ``<variable>``
    Variable name of an internal cache variable to store the result of the
    check, with boolean true for success and boolean false for failure.

  ``FAIL_REGEX <regexes>...``
    If one or more regular expression patterns are provided, then failure is
    determined by checking if anything in the compiler output matches any of
    the specified regular expressions.

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

The following example demonstrates how to check whether the C++ compiler
supports a specific language feature.  In this case, the check verifies if
the compiler supports ``C++11`` lambda expressions.  The result is stored
in the internal cache variable ``HAVE_CXX11_LAMBDAS``:

.. code-block:: cmake

  include(CheckCXXSourceCompiles)

  check_cxx_source_compiles("
    int main()
    {
      auto lambda = []() { return 42; };
      return lambda();
    }
  " HAVE_CXX11_LAMBDAS)

See Also
^^^^^^^^

* The :module:`CheckSourceCompiles` module for a more general command to
  check whether source can be built.
* The :module:`CheckSourceRuns` module to check whether source can be built
  and run.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceCompiles)

macro(CHECK_CXX_SOURCE_COMPILES SOURCE VAR)
  cmake_check_source_compiles(CXX "${SOURCE}" ${VAR} ${ARGN})
endmacro()
