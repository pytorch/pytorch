# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


#[=======================================================================[.rst:
CheckSourceRuns
-------------------

.. versionadded:: 3.19

This module provides a command to check whether a source code can be built
and run.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckSourceRuns)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_source_runs

  Checks once whether the given source code compiles and links into an
  executable that can subsequently be run:

  .. code-block:: cmake

    check_source_runs(<lang> <code> <variable> [SRC_EXT <extension>])

  This command checks once that the ``<lang>`` source code supplied in
  ``<code>`` can be built, linked as an executable, and then run.  The
  result of the check is stored in the internal cache variable specified by
  ``<variable>``.

  The arguments are:

  ``<lang>``
    The programming language of the source ``<code>`` to check.  Supported
    languages are: ``C``, ``CXX``, ``CUDA``, ``Fortran``, ``HIP``, ``OBJC``,
    and ``OBJCXX``.

    .. versionadded:: 3.21
      Support for ``HIP`` language.

  ``<code>``
    The source code to be tested.  It must contain a valid source program.
    For example, it must contain at least a ``main()`` function (in C/C++),
    or a ``program`` unit (in Fortran).

  ``<variable>``
    Name of the internal cache variable with the result of the check.  If
    the code builds and runs with exit code ``0``, success is indicated by
    a boolean true value.  Failure to build or run is indicated by a boolean
    false value, such as an empty string or an error message.

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

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

The following example demonstrates how to use this module to check whether
the C source code is supported and operational at runtime.  The result of
the check is stored in the internal cache variable ``HAVE_NORETURN``.

.. code-block:: cmake

  include(CheckSourceRuns)

  check_source_runs(C "
    #include <stdlib.h>
    #include <stdnoreturn.h>
    noreturn void f(){ exit(0); }
    int main(void) { f(); return 1; }
  " HAVE_NORETURN)

Example: Checking Fortran Code
""""""""""""""""""""""""""""""

Checking if Fortran source code runs successfully:

.. code-block:: cmake

  include(CheckSourceRuns)

  check_source_runs(Fortran "
    program test
    real :: x[*]
    call co_sum(x)
    end program
  " HAVE_COARRAY)

Example: Checking C++ Code With Bracket Argument
""""""""""""""""""""""""""""""""""""""""""""""""

The following example demonstrates how to check whether the C++ standard
library is functional and ``std::vector`` works at runtime.  If the source
compiles, links, and runs successfully, internal cache variable
``HAVE_WORKING_STD_VECTOR`` will be set to boolean true value.  Code is
supplied using :ref:`Bracket Argument` for easier embedded quotes handling:

.. code-block:: cmake
  :force:

  include(CheckSourceRuns)

  check_source_runs(CXX [[
    #include <iostream>
    #include <vector>

    int main()
    {
      std::vector<int> v = {1, 2, 3};
      if (v.size() != 3) return 1;
      std::cout << "Vector works correctly." << std::endl;
      return 0;
    }
  ]] HAVE_WORKING_STD_VECTOR)

Example: Isolated Check
"""""""""""""""""""""""

In the following example, this module is used in combination with the
:module:`CMakePushCheckState` module to modify required compile definitions
and libraries when checking whether the C function ``sched_getcpu()`` is
supported and operational at runtime.  For example, on some systems, the
``sched_getcpu()`` function may be available at compile time but not actually
implemented by the kernel.  In such cases, it returns ``-1`` and sets
``errno`` to ``ENOSYS``.  This check verifies that ``sched_getcpu()`` runs
successfully and stores a boolean result in the internal cache variable
``HAVE_SCHED_GETCPU``.

.. code-block:: cmake

  include(CheckSourceRuns)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)

    if(CMAKE_SYSTEM_NAME STREQUAL "Haiku")
      set(CMAKE_REQUIRED_LIBRARIES gnu)
    endif()

    check_source_runs(C "
      #include <sched.h>
      int main(void)
      {
        if (sched_getcpu() == -1) {
          return 1;
        }
        return 0;
      }
    " HAVE_SCHED_GETCPU)
  cmake_pop_check_state()

See Also
^^^^^^^^

* The :module:`CheckSourceCompiles` module to check whether a source code
  can be built.
#]=======================================================================]

include_guard(GLOBAL)
include(Internal/CheckSourceRuns)

function(CHECK_SOURCE_RUNS _lang _source _var)
  cmake_check_source_runs(${_lang} "${_source}" ${_var} ${ARGN})
endfunction()
