# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


#[=======================================================================[.rst:
CheckSourceCompiles
----------------------

.. versionadded:: 3.19

This module provides a command that checks whether a source code can be
built for a given language.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckSourceCompiles)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_source_compiles

  Checks once whether the given source code can be built for the given
  language:

  .. code-block:: cmake

    check_source_compiles(
      <lang>
      <code>
      <variable>
      [FAIL_REGEX <regexes>...]
      [SRC_EXT <extension>]
    )

  This command checks once that the source supplied in ``<code>`` can be
  compiled (and linked into an executable) for code language ``<lang>``.
  The result of the check is stored in the internal cache variable specified
  by ``<variable>``.

  The arguments are:

  ``<lang>``
    Language of the source code to check.  Supported languages are:
    ``C``, ``CXX``, ``CUDA``, ``Fortran``, ``HIP``, ``ISPC``, ``OBJC``,
    ``OBJCXX``, and ``Swift``.

    .. versionadded:: 3.21
      Support for ``HIP`` language.

    .. versionadded:: 3.26
      Support for ``Swift`` language.

  ``<code>``
    The source code to check.  This must be an entire program, as written
    in a file containing the body block.  All symbols used in the source code
    are expected to be declared as usual in their corresponding headers.

  ``<variable>``
    Variable name of an internal cache variable to store the result of the
    check, with boolean true for success and boolean false for failure.

  ``FAIL_REGEX <regexes>...``
    If one or more regular expression patterns are provided, then failure is
    determined by checking if anything in the compiler output matches any of
    the specified regular expressions.

  ``SRC_EXT <extension>``
    By default, the internal test source file used for the check will be
    given a file extension that matches the requested language (e.g., ``.c``
    for C, ``.cxx`` for C++, ``.F90`` for Fortran, etc.).  This option can
    be used to override this with the ``.<extension>`` instead.

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

Example: Basic Usage
""""""""""""""""""""

The following example demonstrates how to check whether the C++ compiler
supports a specific language feature using this module.  In this case, the
check verifies if the compiler supports ``C++11`` lambda expressions.  The
result is stored in the internal cache variable ``HAVE_CXX11_LAMBDAS``:

.. code-block:: cmake

  include(CheckSourceCompiles)

  check_source_compiles(CXX "
    int main()
    {
      auto lambda = []() { return 42; };
      return lambda();
    }
  " HAVE_CXX11_LAMBDAS)

Example: Checking Code With Bracket Argument
""""""""""""""""""""""""""""""""""""""""""""

The following example shows how to check whether the C compiler supports the
``noreturn`` attribute.  Code is supplied using the :ref:`Bracket Argument`
for easier embedded quotes handling:

.. code-block:: cmake
  :force:

  include(CheckSourceCompiles)

  check_source_compiles(C [[
    #if !__has_c_attribute(noreturn)
    #  error "No noreturn attribute"
    #endif
    int main(void) { return 0; }
  ]] HAVE_NORETURN)

Example: Performing a Check Without Linking
"""""""""""""""""""""""""""""""""""""""""""

In the following example, this module is used to perform a compile-only
check of Fortran source code, whether the compiler supports the ``pure``
procedure attribute:

.. code-block:: cmake

  include(CheckSourceCompiles)

  block()
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

    check_source_compiles(
      Fortran
      "pure subroutine foo()
      end subroutine"
      HAVE_PURE
    )
  endblock()

Example: Isolated Check
"""""""""""""""""""""""

In the following example, this module is used in combination with the
:module:`CMakePushCheckState` module to modify required libraries when
checking whether the PostgreSQL ``PGVerbosity`` enum contains
``PQERRORS_SQLSTATE`` (available as of PostgreSQL version 12):

.. code-block:: cmake

  include(CheckSourceCompiles)
  include(CMakePushCheckState)

  find_package(PostgreSQL)

  if(TARGET PostgreSQL::PostgreSQL)
    cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_LIBRARIES PostgreSQL::PostgreSQL)

      check_source_compiles(C "
        #include <libpq-fe.h>
        int main(void)
        {
          PGVerbosity e = PQERRORS_SQLSTATE;
          (void)e;
          return 0;
        }
      " HAVE_PQERRORS_SQLSTATE)
    cmake_pop_check_state()
  endif()

See Also
^^^^^^^^

* The :module:`CheckSourceRuns` module to check whether the source code can
  be built and also run.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceCompiles)

function(CHECK_SOURCE_COMPILES _lang _source _var)
  cmake_check_source_compiles(${_lang} "${_source}" ${_var} ${ARGN})
endfunction()
