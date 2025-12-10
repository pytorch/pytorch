# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckCXXSymbolExists
--------------------

This module provides a command to check whether a C++ symbol exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckCXXSymbolExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_cxx_symbol_exists

  Checks once whether a symbol exists as a function, variable, or preprocessor
  macro in C++:

  .. code-block:: cmake

    check_cxx_symbol_exists(<symbol> <headers> <variable>)

  This command checks whether the ``<symbol>`` is available after including
  the specified header file(s) ``<headers>``, and stores the result in the
  internal cache variable ``<variable>``.  Multiple header files can be
  specified in one argument as a string using a
  :ref:`semicolon-separated list <CMake Language Lists>`.

  If the header files define the symbol as a macro, it is considered
  available and assumed to work.  If the symbol is declared as a function
  or variable, the check also ensures that it links successfully
  (i.e., the symbol must exist in a linked library or object file).

  Symbols that are types, enum values, or C++ templates are not
  recognized.  For those, consider using the :module:`CheckTypeSize` or
  :module:`CheckSourceCompiles` module instead.

  This command is intended to check symbols as they appear in C++.  For C
  symbols, use the :module:`CheckSymbolExists` module instead.

  .. note::

    This command is unreliable for symbols that are (potentially) overloaded
    functions.  Since there is no reliable way to predict whether
    a given function in the system environment may be defined as an
    overloaded function or may be an overloaded function on other systems
    or will become so in the future, it is generally advised to use the
    :module:`CheckSourceCompiles` module for checking any function symbol
    (unless it is certain the checked function is not overloaded on other
    systems or will not be so in the future).

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

The following example demonstrates how to check for the presence of a
preprocessor macro ``SEEK_SET`` and the C++ function ``std::fopen()`` from
the ``<cstdio>`` header using this module:

.. code-block:: cmake

  include(CheckCXXSymbolExists)

  # Check for macro SEEK_SET
  check_cxx_symbol_exists(SEEK_SET "cstdio" HAVE_SEEK_SET)

  # Check for function std::fopen
  check_cxx_symbol_exists(std::fopen "cstdio" HAVE_STD_FOPEN)

See Also
^^^^^^^^

* The :module:`CheckSymbolExists` module to check whether a C symbol exists.
#]=======================================================================]

include_guard(GLOBAL)
include(CheckSymbolExists)

macro(CHECK_CXX_SYMBOL_EXISTS SYMBOL FILES VARIABLE)
  __CHECK_SYMBOL_EXISTS_IMPL(CheckSymbolExists.cxx "${SYMBOL}" "${FILES}" "${VARIABLE}" )
endmacro()
