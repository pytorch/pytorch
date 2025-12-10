# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckSymbolExists
-----------------

This module provides a command to check whether a C symbol exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckSymbolExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_symbol_exists

  Checks once whether a symbol exists as a function, variable, or preprocessor
  macro in C:

  .. code-block:: cmake

    check_symbol_exists(<symbol> <headers> <variable>)

  This command checks whether the ``<symbol>`` is available after including
  the specified header file(s) ``<headers>``, and stores the result in the
  internal cache variable ``<variable>``.  Multiple header files can be
  specified in one argument as a string using a
  :ref:`semicolon-separated list <CMake Language Lists>`.

  If the header files define the symbol as a macro, it is considered
  available and assumed to work.  If the symbol is declared as a function
  or variable, the check also ensures that it links successfully
  (i.e., the symbol must exist in a linked library or object file).
  Compiler intrinsics may not be detected, as they are not always linkable
  or explicitly declared in headers.

  Symbols that are types, enum values, or compiler intrinsics are not
  recognized.  For those, consider using the :module:`CheckTypeSize` or
  :module:`CheckSourceCompiles` module instead.

  This command is intended to check symbols as they appear in C.  For C++
  symbols, use the :module:`CheckCXXSymbolExists` module instead.

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
preprocessor macro ``SEEK_SET`` and the C function ``fopen()`` from
the ``<stdio.h>`` header using this module:

.. code-block:: cmake

  include(CheckSymbolExists)

  # Check for macro SEEK_SET
  check_symbol_exists(SEEK_SET "stdio.h" HAVE_SEEK_SET)

  # Check for function fopen
  check_symbol_exists(fopen "stdio.h" HAVE_FOPEN)

See Also
^^^^^^^^

* The :module:`CheckCXXSymbolExists` module to check whether a C++ symbol
  exists.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_SYMBOL_EXISTS SYMBOL FILES VARIABLE)
  if(CMAKE_C_COMPILER_LOADED)
    __CHECK_SYMBOL_EXISTS_FILTER_FLAGS(C)
    __CHECK_SYMBOL_EXISTS_IMPL(CheckSymbolExists.c "${SYMBOL}" "${FILES}" "${VARIABLE}" )
    __CHECK_SYMBOL_EXISTS_RESTORE_FLAGS(C)
  elseif(CMAKE_CXX_COMPILER_LOADED)
    __CHECK_SYMBOL_EXISTS_FILTER_FLAGS(CXX)
    __CHECK_SYMBOL_EXISTS_IMPL(CheckSymbolExists.cxx "${SYMBOL}" "${FILES}" "${VARIABLE}" )
    __CHECK_SYMBOL_EXISTS_RESTORE_FLAGS(CXX)
  else()
    message(FATAL_ERROR "CHECK_SYMBOL_EXISTS needs either C or CXX language enabled")
  endif()
endmacro()

macro(__CHECK_SYMBOL_EXISTS_FILTER_FLAGS LANG)
    if(CMAKE_TRY_COMPILE_CONFIGURATION)
      string(TOUPPER "${CMAKE_TRY_COMPILE_CONFIGURATION}" _tc_config)
    else()
      set(_tc_config "DEBUG")
    endif()
    foreach(v CMAKE_${LANG}_FLAGS CMAKE_${LANG}_FLAGS_${_tc_config})
      set(__${v}_SAVED "${${v}}")
      string(REGEX REPLACE "(^| )-Werror([= ][^-][^ ]*)?( |$)" " " ${v} "${${v}}")
      string(REGEX REPLACE "(^| )-pedantic-errors( |$)" " " ${v} "${${v}}")
    endforeach()
endmacro()

macro(__CHECK_SYMBOL_EXISTS_RESTORE_FLAGS LANG)
    if(CMAKE_TRY_COMPILE_CONFIGURATION)
      string(TOUPPER "${CMAKE_TRY_COMPILE_CONFIGURATION}" _tc_config)
    else()
      set(_tc_config "DEBUG")
    endif()
    foreach(v CMAKE_${LANG}_FLAGS CMAKE_${LANG}_FLAGS_${_tc_config})
      set(${v} "${__${v}_SAVED}")
      unset(__${v}_SAVED)
    endforeach()
endmacro()

macro(__CHECK_SYMBOL_EXISTS_IMPL SOURCEFILE SYMBOL FILES VARIABLE)
  if(NOT DEFINED "${VARIABLE}" OR "x${${VARIABLE}}" STREQUAL "x${VARIABLE}")
    set(_CSE_SOURCE "/* */\n")
    set(MACRO_CHECK_SYMBOL_EXISTS_FLAGS ${CMAKE_REQUIRED_FLAGS})
    if(CMAKE_REQUIRED_LINK_OPTIONS)
      set(CHECK_SYMBOL_EXISTS_LINK_OPTIONS
        LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    else()
      set(CHECK_SYMBOL_EXISTS_LINK_OPTIONS)
    endif()
    if(CMAKE_REQUIRED_LIBRARIES)
      set(CHECK_SYMBOL_EXISTS_LIBS
        LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CHECK_SYMBOL_EXISTS_LIBS)
    endif()
    if(CMAKE_REQUIRED_INCLUDES)
      set(CMAKE_SYMBOL_EXISTS_INCLUDES
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CMAKE_SYMBOL_EXISTS_INCLUDES)
    endif()

    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CSE_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CSE_LINK_DIRECTORIES)
    endif()
    foreach(FILE ${FILES})
      string(APPEND _CSE_SOURCE
        "#include <${FILE}>\n")
    endforeach()
    string(APPEND _CSE_SOURCE "
int main(int argc, char** argv)
{
  (void)argv;")
    set(_CSE_CHECK_NON_MACRO "return ((int*)(&${SYMBOL}))[argc];")
    if("${SYMBOL}" MATCHES "^[a-zA-Z_][a-zA-Z0-9_]*$")
      # The SYMBOL has a legal macro name.  Test whether it exists as a macro.
      string(APPEND _CSE_SOURCE "
#ifndef ${SYMBOL}
  ${_CSE_CHECK_NON_MACRO}
#else
  (void)argc;
  return 0;
#endif")
    else()
      # The SYMBOL cannot be a macro (e.g., a template function).
      string(APPEND _CSE_SOURCE "
  ${_CSE_CHECK_NON_MACRO}")
    endif()
    string(APPEND _CSE_SOURCE "
}\n")
    unset(_CSE_CHECK_NON_MACRO)

    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${SYMBOL}")
    endif()
    try_compile(${VARIABLE}
      SOURCE_FROM_VAR "${SOURCEFILE}" _CSE_SOURCE
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${CHECK_SYMBOL_EXISTS_LINK_OPTIONS}
      ${CHECK_SYMBOL_EXISTS_LIBS}
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_SYMBOL_EXISTS_FLAGS}
      "${CMAKE_SYMBOL_EXISTS_INCLUDES}"
      "${_CSE_LINK_DIRECTORIES}"
      )
    unset(_CSE_LINK_DIRECTORIES)
    if(${VARIABLE})
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
      set(${VARIABLE} 1 CACHE INTERNAL "Have symbol ${SYMBOL}")
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
      set(${VARIABLE} "" CACHE INTERNAL "Have symbol ${SYMBOL}")
    endif()
    unset(_CSE_SOURCE)
  endif()
endmacro()
