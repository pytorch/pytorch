# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckPrototypeDefinition
------------------------

This module provides a command to check if a C function has the expected
prototype.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckPrototypeDefinition)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_prototype_definition

  Checks if a C function has the expected prototype:

  .. code-block:: cmake

    check_prototype_definition(<function> <prototype> <return> <headers> <variable>)

  ``<function>``
    The name of the function whose prototype is being checked.
  ``<prototype>``
    The expected prototype of the function, provided as a string.
  ``<return>``
    The return value of the function.  This will be used as a return value in
    the function definition body of the generated test program to verify that
    the function's return type matches the expected type.
  ``<headers>``
    A :ref:`semicolon-separated list <CMake Language Lists>` of header file
    names required for checking the function prototype.
  ``<variable>``
    The name of the variable to store the check result.  This variable will be
    created as an internal cache variable.

  This command generates a test program and verifies that it builds without
  errors.  The generated test program includes specified ``<headers>``, defines
  a function with given literal ``<prototype>`` and ``<return>`` value and
  then uses the specified ``<function>``.  The simplified test program can be
  illustrated as:

  .. code-block:: c

    #include <headers>
    // ...
    <prototype> { return <return>; }
    int main(...) { ...<function>()... }

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

Checking if the ``getpwent_r()`` function on Solaris/illumos systems has the
expected prototype:

.. code-block:: cmake

  include(CheckPrototypeDefinition)

  check_prototype_definition(
    getpwent_r
    "struct passwd *getpwent_r(struct passwd *src, char *buf, int buflen)"
    "NULL"
    "unistd.h;pwd.h"
    HAVE_SOLARIS_GETPWENT_R
  )
#]=======================================================================]

include_guard(GLOBAL)

function(check_prototype_definition _FUNCTION _PROTOTYPE _RETURN _HEADER _VARIABLE)

  if (NOT DEFINED ${_VARIABLE})
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Checking prototype ${_FUNCTION} for ${_VARIABLE}")
    endif()
    set(CHECK_PROTOTYPE_DEFINITION_CONTENT "/* */\n")

    set(CHECK_PROTOTYPE_DEFINITION_FLAGS ${CMAKE_REQUIRED_FLAGS})
    if (CMAKE_REQUIRED_LINK_OPTIONS)
      set(CHECK_PROTOTYPE_DEFINITION_LINK_OPTIONS
        LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    else()
      set(CHECK_PROTOTYPE_DEFINITION_LINK_OPTIONS)
    endif()
    if (CMAKE_REQUIRED_LIBRARIES)
      set(CHECK_PROTOTYPE_DEFINITION_LIBS
        LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    else()
      set(CHECK_PROTOTYPE_DEFINITION_LIBS)
    endif()
    if (CMAKE_REQUIRED_INCLUDES)
      set(CMAKE_SYMBOL_EXISTS_INCLUDES
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CMAKE_SYMBOL_EXISTS_INCLUDES)
    endif()

    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CPD_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CPD_LINK_DIRECTORIES)
    endif()

    foreach(_FILE ${_HEADER})
      string(APPEND CHECK_PROTOTYPE_DEFINITION_HEADER
        "#include <${_FILE}>\n")
    endforeach()

    set(CHECK_PROTOTYPE_DEFINITION_SYMBOL ${_FUNCTION})
    set(CHECK_PROTOTYPE_DEFINITION_PROTO ${_PROTOTYPE})
    set(CHECK_PROTOTYPE_DEFINITION_RETURN ${_RETURN})

    file(READ ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CheckPrototypeDefinition.c.in _SOURCE)
    string(CONFIGURE "${_SOURCE}" _SOURCE @ONLY)

    try_compile(${_VARIABLE}
      SOURCE_FROM_VAR CheckPrototypeDefinition.c _SOURCE
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${CHECK_PROTOTYPE_DEFINITION_LINK_OPTIONS}
      ${CHECK_PROTOTYPE_DEFINITION_LIBS}
      CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${CHECK_PROTOTYPE_DEFINITION_FLAGS}
      "${CMAKE_SYMBOL_EXISTS_INCLUDES}"
      "${_CPD_LINK_DIRECTORIES}"
      )
    unset(_CPD_LINK_DIRECTORIES)

    if (${_VARIABLE})
      set(${_VARIABLE} 1 CACHE INTERNAL "Have correct prototype for ${_FUNCTION}")
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "True")
      endif()
    else ()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "False")
      endif()
      set(${_VARIABLE} 0 CACHE INTERNAL "Have correct prototype for ${_FUNCTION}")
    endif ()
  endif()

endfunction()
