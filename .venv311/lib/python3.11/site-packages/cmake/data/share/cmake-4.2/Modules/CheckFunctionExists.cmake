# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckFunctionExists
-------------------

This module provides a command to check whether a C function exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckFunctionExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_function_exists

  Checks once whether a C function can be linked from system libraries:

  .. code-block:: cmake

    check_function_exists(<function> <variable>)

  This command checks whether the ``<function>`` is provided by libraries
  on the system, and stores the result in an internal cache variable
  ``<variable>``.

  .. note::

    Prefer using :module:`CheckSymbolExists` or :module:`CheckSourceCompiles`
    instead of this command, for the following reasons:

    * ``check_function_exists()`` can't detect functions that are inlined
      in headers or defined as preprocessor macros.

    * ``check_function_exists()`` can't detect anything in the 32-bit
      versions of the Win32 API, because of a mismatch in calling conventions.

    * ``check_function_exists()`` only verifies linking, it does not verify
      that the function is declared in system headers.

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

In the following example, a check is performed to determine whether the
linker sees the C function ``fopen()``, and the result is stored in the
``HAVE_FOPEN`` internal cache variable:

.. code-block:: cmake

  include(CheckFunctionExists)

  check_function_exists(fopen HAVE_FOPEN)

Example: Missing Declaration
""""""""""""""""""""""""""""

As noted above, the :module:`CheckSymbolExists` module is preferred for
checking C functions, since it also verifies whether the function is
declared or defined as a macro.  In the following example, this module is
used to check an edge case where a function may not be declared in system
headers.  For instance, on macOS, the ``fdatasync()`` function may be
available in the C library, but its declaration is not provided in the
``unistd.h`` system header.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CheckFunctionExists)
  include(CheckSymbolExists)

  check_symbol_exists(fdatasync "unistd.h" HAVE_FDATASYNC)

  # Check if fdatasync() is available in the C library.
  if(NOT HAVE_FDATASYNC)
    check_function_exists(fdatasync HAVE_FDATASYNC_WITHOUT_DECL)
  endif()

In such a case, the project can provide its own declaration if missing:

.. code-block:: c
  :caption: ``example.c``

  #ifdef HAVE_FDATASYNC_WITHOUT_DECL
    extern int fdatasync(int);
  #endif

See Also
^^^^^^^^

* The :module:`CheckSymbolExists` module to check whether a C symbol exists.
* The :module:`CheckSourceCompiles` module to check whether a source code
  can be compiled.
* The :module:`CheckFortranFunctionExists` module to check whether a
  Fortran function exists.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_FUNCTION_EXISTS FUNCTION VARIABLE)
  if(NOT DEFINED "${VARIABLE}" OR "x${${VARIABLE}}" STREQUAL "x${VARIABLE}")
    set(MACRO_CHECK_FUNCTION_DEFINITIONS
      "-DCHECK_FUNCTION_EXISTS=${FUNCTION} ${CMAKE_REQUIRED_FLAGS}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${FUNCTION}")
    endif()
    if(CMAKE_REQUIRED_LINK_OPTIONS)
      set(CHECK_FUNCTION_EXISTS_ADD_LINK_OPTIONS
        LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    else()
      set(CHECK_FUNCTION_EXISTS_ADD_LINK_OPTIONS)
    endif()
    if(CMAKE_REQUIRED_LIBRARIES)
      set(CHECK_FUNCTION_EXISTS_ADD_LIBRARIES
        LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CHECK_FUNCTION_EXISTS_ADD_LIBRARIES)
    endif()
    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CFE_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CFE_LINK_DIRECTORIES)
    endif()
    if(CMAKE_REQUIRED_INCLUDES)
      set(CHECK_FUNCTION_EXISTS_ADD_INCLUDES
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CHECK_FUNCTION_EXISTS_ADD_INCLUDES)
    endif()

    if(CMAKE_C_COMPILER_LOADED)
      set(_cfe_source CheckFunctionExists.c)
    elseif(CMAKE_CXX_COMPILER_LOADED)
      set(_cfe_source CheckFunctionExists.cxx)
    else()
      message(FATAL_ERROR "CHECK_FUNCTION_EXISTS needs either C or CXX language enabled")
    endif()

    try_compile(${VARIABLE}
      SOURCE_FROM_FILE "${_cfe_source}" "${CMAKE_ROOT}/Modules/CheckFunctionExists.c"
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${CHECK_FUNCTION_EXISTS_ADD_LINK_OPTIONS}
      ${CHECK_FUNCTION_EXISTS_ADD_LIBRARIES}
      CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_FUNCTION_DEFINITIONS}
      "${CHECK_FUNCTION_EXISTS_ADD_INCLUDES}"
      "${_CFE_LINK_DIRECTORIES}"
      )
    unset(_cfe_source)
    unset(_CFE_LINK_DIRECTORIES)

    if(${VARIABLE})
      set(${VARIABLE} 1 CACHE INTERNAL "Have function ${FUNCTION}")
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
      set(${VARIABLE} "" CACHE INTERNAL "Have function ${FUNCTION}")
    endif()
  endif()
endmacro()
