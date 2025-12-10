# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckIncludeFiles
-----------------

This module provides a command to check one or more C/C++ header files.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckIncludeFiles)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_include_files

  Checks once whether one or more header files exist and can be included
  together in C or C++ code:

  .. code-block:: cmake

    check_include_files(<includes> <variable> [LANGUAGE <language>])

  .. rubric:: The arguments are:

  ``<includes>``
    A :ref:`semicolon-separated list <CMake Language Lists>` of header
    files to be checked.

  ``<variable>``
    The name of the variable to store the result of the check.  This
    variable will be created as an internal cache variable.

  ``LANGUAGE <language>``
    .. versionadded:: 3.11

    If set, the specified ``<language>`` compiler will be used to perform
    the check.  Acceptable values are ``C`` and ``CXX``.  If this option is
    not given, the C compiler will be used if enabled.  If the C compiler
    is not enabled, the C++ compiler will be used if enabled.

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

  .. versionadded:: 3.12
    The ``CMAKE_REQUIRED_LIBRARIES`` variable, if policy :policy:`CMP0075` is
    set to ``NEW``.

Examples
^^^^^^^^

Checking one or more C headers and storing the check result in cache
variables:

.. code-block:: cmake

  include(CheckIncludeFiles)

  check_include_files(sys/socket.h HAVE_SYS_SOCKET_H)

  if(HAVE_SYS_SOCKET_H)
    # The <net/if.h> header on Darwin and BSD-like systems is not self-contained
    # and also requires <sys/socket.h>
    check_include_files("sys/socket.h;net/if.h" HAVE_NET_IF_H)
  else()
    check_include_files(net/if.h HAVE_NET_IF_H)
  endif()

The ``LANGUAGE`` option can be used to specify which compiler to use.  For
example, checking multiple ``C++`` headers, when both ``C`` and ``CXX``
languages are enabled in the project:

.. code-block:: cmake

  include(CheckIncludeFiles)

  check_include_files("header_1.hpp;header_2.hpp" HAVE_HEADERS LANGUAGE CXX)

See Also
^^^^^^^^

* The :module:`CheckIncludeFile` module to check for a single C header.
* The :module:`CheckIncludeFileCXX` module to check for a single C++ header.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_INCLUDE_FILES INCLUDE VARIABLE)
  if(NOT DEFINED "${VARIABLE}")
    set(_src_content "/* */\n")

    if("x${ARGN}" STREQUAL "x")
       if(CMAKE_C_COMPILER_LOADED)
         set(_lang C)
       elseif(CMAKE_CXX_COMPILER_LOADED)
         set(_lang CXX)
       else()
         message(FATAL_ERROR "CHECK_INCLUDE_FILES needs either C or CXX language enabled.\n")
       endif()
    elseif("x${ARGN}" MATCHES "^xLANGUAGE;([a-zA-Z]+)$")
      set(_lang "${CMAKE_MATCH_1}")
    elseif("x${ARGN}" MATCHES "^xLANGUAGE$")
      message(FATAL_ERROR "No languages listed for LANGUAGE option.\nSupported languages: C, CXX.\n")
    else()
      message(FATAL_ERROR "Unknown arguments:\n  ${ARGN}\n")
    endif()

    string(MAKE_C_IDENTIFIER ${VARIABLE} _variable_escaped)
    if(_lang STREQUAL "C")
      set(src ${_variable_escaped}.c)
    elseif(_lang STREQUAL "CXX")
      set(src ${_variable_escaped}.cpp)
    else()
      message(FATAL_ERROR "Unknown language:\n  ${_lang}\nSupported languages: C, CXX.\n")
    endif()

    if(CMAKE_REQUIRED_INCLUDES)
      set(CHECK_INCLUDE_FILES_INCLUDE_DIRS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CHECK_INCLUDE_FILES_INCLUDE_DIRS)
    endif()
    set(CHECK_INCLUDE_FILES_CONTENT "/* */\n")
    set(MACRO_CHECK_INCLUDE_FILES_FLAGS ${CMAKE_REQUIRED_FLAGS})
    foreach(FILE ${INCLUDE})
      string(APPEND _src_content
        "#include <${FILE}>\n")
    endforeach()
    string(APPEND _src_content
      "\n\nint main(void){return 0;}\n")

    set(_INCLUDE ${INCLUDE}) # remove empty elements
    if("${_INCLUDE}" MATCHES "^([^;]+);.+;([^;]+)$")
      list(LENGTH _INCLUDE _INCLUDE_LEN)
      set(_description "${_INCLUDE_LEN} include files ${CMAKE_MATCH_1}, ..., ${CMAKE_MATCH_2}")
    elseif("${_INCLUDE}" MATCHES "^([^;]+);([^;]+)$")
      set(_description "include files ${CMAKE_MATCH_1}, ${CMAKE_MATCH_2}")
    else()
      set(_description "include file ${_INCLUDE}")
    endif()

    set(_CIF_LINK_OPTIONS)
    if(CMAKE_REQUIRED_LINK_OPTIONS)
      set(_CIF_LINK_OPTIONS LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    endif()

    set(_CIF_LINK_LIBRARIES "")
    if(CMAKE_REQUIRED_LIBRARIES)
      cmake_policy(GET CMP0075 _CIF_CMP0075
        PARENT_SCOPE # undocumented, do not use outside of CMake
        )
      if("x${_CIF_CMP0075}x" STREQUAL "xNEWx")
        set(_CIF_LINK_LIBRARIES LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
      elseif("x${_CIF_CMP0075}x" STREQUAL "xOLDx")
      elseif(NOT _CIF_CMP0075_WARNED)
        set(_CIF_CMP0075_WARNED 1)
        message(AUTHOR_WARNING
          "Policy CMP0075 is not set: Include file check macros honor CMAKE_REQUIRED_LIBRARIES.  "
          "Run \"cmake --help-policy CMP0075\" for policy details.  "
          "Use the cmake_policy command to set the policy and suppress this warning."
          "\n"
          "CMAKE_REQUIRED_LIBRARIES is set to:\n"
          "  ${CMAKE_REQUIRED_LIBRARIES}\n"
          "For compatibility with CMake 3.11 and below this check is ignoring it."
          )
      endif()
      unset(_CIF_CMP0075)
    endif()

    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CIF_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CIF_LINK_DIRECTORIES)
    endif()

    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${_description}")
    endif()
    try_compile(${VARIABLE}
      SOURCE_FROM_VAR "${src}" _src_content
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${_CIF_LINK_OPTIONS}
      ${_CIF_LINK_LIBRARIES}
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_INCLUDE_FILES_FLAGS}
      "${CHECK_INCLUDE_FILES_INCLUDE_DIRS}"
      "${_CIF_LINK_DIRECTORIES}"
      )
    unset(_CIF_LINK_OPTIONS)
    unset(_CIF_LINK_LIBRARIES)
    unset(_CIF_LINK_DIRECTORIES)
    if(${VARIABLE})
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
      set(${VARIABLE} 1 CACHE INTERNAL "Have include ${INCLUDE}")
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
      set(${VARIABLE} "" CACHE INTERNAL "Have includes ${INCLUDE}")
    endif()
  endif()
endmacro()
