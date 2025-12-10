# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckVariableExists
-------------------

This module provides a command to check whether a C variable exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckVariableExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_variable_exists

  Checks once if a C variable exists:

  .. code-block:: cmake

    check_variable_exists(<var> <variable>)

  This command attempts to compile and link a test C program that references
  the specified C variable ``<var>``.  A boolean result of whether
  the check was successful is stored in an internal cache variable
  ``<variable>``.

  .. note::

    Prefer using :module:`CheckSymbolExists` or :module:`CheckSourceCompiles`
    instead of this command for more robust detection.  This command performs
    a link-only check and doesn't detect whether a variable is also declared
    in system or library headers.  Neither can it detect variables that might
    be defined as preprocessor macros.

  .. rubric:: Variables Affecting the Check

  The following variables may be set before calling this command to modify
  the way the check is run:

  .. include:: /module/include/CMAKE_REQUIRED_FLAGS.rst

  .. include:: /module/include/CMAKE_REQUIRED_DEFINITIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

In the following example, a check is performed whether the linker sees the
C variable ``tzname`` and stores the check result in the
``PROJECT_HAVE_TZNAME`` internal cache variable:

.. code-block:: cmake

  include(CheckVariableExists)

  check_variable_exists(tzname PROJECT_HAVE_TZNAME)

Example: Isolated Check With Linked Libraries
"""""""""""""""""""""""""""""""""""""""""""""

In the following example, this module is used in combination with the
:module:`CMakePushCheckState` module to link additional required library
using the ``CMAKE_REQUIRED_LIBRARIES`` variable.  For example, in a find
module, to check whether the Net-SNMP library has the
``usmHMAC192SHA256AuthProtocol`` array:

.. code-block:: cmake

  include(CheckVariableExists)
  include(CMakePushCheckState)

  find_library(SNMP_LIBRARY NAMES netsnmp)

  if(SNMP_LIBRARY)
    cmake_push_check_state(RESET)

    set(CMAKE_REQUIRED_LIBRARIES ${SNMP_LIBRARY})

    check_variable_exists(usmHMAC192SHA256AuthProtocol SNMP_HAVE_SHA256)

    cmake_pop_check_state()
  endif()

See Also
^^^^^^^^

* The :module:`CheckSymbolExists` module to check whether a C symbol exists.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_VARIABLE_EXISTS VAR VARIABLE)
  if(NOT DEFINED "${VARIABLE}")
    set(MACRO_CHECK_VARIABLE_DEFINITIONS
      "-DCHECK_VARIABLE_EXISTS=${VAR} ${CMAKE_REQUIRED_FLAGS}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${VAR}")
    endif()
    if(CMAKE_REQUIRED_LINK_OPTIONS)
      set(CHECK_VARIABLE_EXISTS_ADD_LINK_OPTIONS
        LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    else()
      set(CHECK_VARIABLE_EXISTS_ADD_LINK_OPTIONS)
    endif()
    if(CMAKE_REQUIRED_LIBRARIES)
      set(CHECK_VARIABLE_EXISTS_ADD_LIBRARIES
        LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CHECK_VARIABLE_EXISTS_ADD_LIBRARIES)
    endif()

    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CVE_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CVE_LINK_DIRECTORIES)
    endif()

    try_compile(${VARIABLE}
      SOURCES ${CMAKE_ROOT}/Modules/CheckVariableExists.c
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${CHECK_VARIABLE_EXISTS_ADD_LINK_OPTIONS}
      ${CHECK_VARIABLE_EXISTS_ADD_LIBRARIES}
      CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_VARIABLE_DEFINITIONS}
      "${_CVE_LINK_DIRECTORIES}"
      )
    unset(_CVE_LINK_DIRECTORIES)
    if(${VARIABLE})
      set(${VARIABLE} 1 CACHE INTERNAL "Have variable ${VAR}")
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
    else()
      set(${VARIABLE} "" CACHE INTERNAL "Have variable ${VAR}")
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
    endif()
  endif()
endmacro()
