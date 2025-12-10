# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckFortranFunctionExists
--------------------------

This module provides a command to check whether a Fortran function exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckFortranFunctionExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_fortran_function_exists

  Checks once whether a Fortran function exists:

  .. code-block:: cmake

    check_fortran_function_exists(<function> <variable>)

  ``<function>``
    The name of the Fortran function.

  ``<variable>``
    The name of the variable in which to store the check result.  This
    variable will be created as an internal cache variable.

  .. note::

    This command does not detect functions provided by Fortran modules.  In
    general, it is recommended to use :module:`CheckSourceCompiles` instead
    to determine whether a Fortran function or subroutine is available.

  .. rubric:: Variables Affecting the Check

  The following variables may be set before calling this command to modify
  the way the check is run:

  .. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

Examples
^^^^^^^^

Example: Isolated Check With Linked Libraries
"""""""""""""""""""""""""""""""""""""""""""""

In the following example, this module is used in combination with the
:module:`CMakePushCheckState` module to temporarily modify the required
linked libraries (via ``CMAKE_REQUIRED_LIBRARIES``) and verify whether the
Fortran function ``dgesv`` is available for linking.  The result is stored
in the internal cache variable ``PROJECT_HAVE_DGESV``:

.. code-block:: cmake

  include(CheckFortranFunctionExists)
  include(CMakePushCheckState)

  find_package(LAPACK)

  if(TARGET LAPACK::LAPACK)
    cmake_push_check_state(RESET)

    set(CMAKE_REQUIRED_LIBRARIES LAPACK::LAPACK)
    check_fortran_function_exists(dgesv PROJECT_HAVE_DGESV)

    cmake_pop_check_state()
  endif()

See Also
^^^^^^^^

* The :module:`CheckFunctionExists` module to check whether a C function
  exists.
* The :module:`CheckSourceCompiles` module to check whether source code
  can be compiled.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_FORTRAN_FUNCTION_EXISTS FUNCTION VARIABLE)
  if(NOT DEFINED ${VARIABLE})
    message(CHECK_START "Looking for Fortran ${FUNCTION}")
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
      set(_CFFE_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CFFE_LINK_DIRECTORIES)
    endif()
    set(__CheckFunction_testFortranCompilerSource
    "
      program TESTFortran
      external ${FUNCTION}
      call ${FUNCTION}()
      end program TESTFortran
    "
    )
    try_compile(${VARIABLE}
      SOURCE_FROM_VAR testFortranCompiler.f __CheckFunction_testFortranCompilerSource
      ${CHECK_FUNCTION_EXISTS_ADD_LINK_OPTIONS}
      ${CHECK_FUNCTION_EXISTS_ADD_LIBRARIES}
      CMAKE_FLAGS
      "${_CFFE_LINK_DIRECTORIES}"
    )
    unset(__CheckFunction_testFortranCompilerSource)
    unset(_CFFE_LINK_DIRECTORIES)
    if(${VARIABLE})
      set(${VARIABLE} 1 CACHE INTERNAL "Have Fortran function ${FUNCTION}")
      message(CHECK_PASS "found")
    else()
      message(CHECK_FAIL "not found")
      set(${VARIABLE} "" CACHE INTERNAL "Have Fortran function ${FUNCTION}")
    endif()
  endif()
endmacro()
