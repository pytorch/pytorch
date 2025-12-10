# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckIncludeFile
----------------

This module provides a command to check C header file.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckIncludeFile)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_include_file

  Checks once whether a header file exists and can be included in C code:

  .. code-block:: cmake

    check_include_file(<include> <variable> [<flags>])

  .. rubric:: The arguments are:

  ``<include>``
    A header file to be checked.

  ``<variable>``
    The name of the variable to store the result of the check.  This
    variable will be created as an internal cache variable.

  ``<flags>``
    (Optional) A :ref:`semicolon-separated list <CMake Language Lists>` of
    additional compilation flags to be added to the check.  Alternatively,
    flags can be also specified with the ``CMAKE_REQUIRED_FLAGS`` variable
    below.

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

Example: Checking C Header
""""""""""""""""""""""""""

Checking whether the C header ``<unistd.h>`` exists and storing the check
result in the ``HAVE_UNISTD_H`` cache variable:

.. code-block:: cmake

  include(CheckIncludeFile)

  check_include_file(unistd.h HAVE_UNISTD_H)

Example: Isolated Check
"""""""""""""""""""""""

In the following example, this module is used in combination with the
:module:`CMakePushCheckState` module to temporarily modify the required
compile definitions (via ``CMAKE_REQUIRED_DEFINITIONS``) and verify whether
the C header ``<ucontext.h>`` is available.  The result is stored
in the internal cache variable ``HAVE_UCONTEXT_H``.

For example, on macOS, the ``ucontext`` API is deprecated, and headers may
be hidden unless certain feature macros are defined.  In particular,
defining ``_XOPEN_SOURCE`` (without a value) can expose the necessary
symbols without enabling broader POSIX or SUS (Single Unix Specification)
features (values 500 or greater).

.. code-block:: cmake

  include(CheckIncludeFile)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      set(CMAKE_REQUIRED_DEFINITIONS -D_XOPEN_SOURCE)
    endif()

    check_include_file(ucontext.h HAVE_UCONTEXT_H)
  cmake_pop_check_state()

See Also
^^^^^^^^

* The :module:`CheckIncludeFileCXX` module to check for single C++ header.
* The :module:`CheckIncludeFiles` module to check for one or more C or
  C++ headers at once.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_INCLUDE_FILE INCLUDE VARIABLE)
  if(NOT DEFINED "${VARIABLE}")
    if(CMAKE_REQUIRED_INCLUDES)
      set(CHECK_INCLUDE_FILE_C_INCLUDE_DIRS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CHECK_INCLUDE_FILE_C_INCLUDE_DIRS)
    endif()
    set(MACRO_CHECK_INCLUDE_FILE_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CHECK_INCLUDE_FILE_VAR ${INCLUDE})
    file(READ ${CMAKE_ROOT}/Modules/CheckIncludeFile.c.in _CIF_SOURCE_CONTENT)
    string(CONFIGURE "${_CIF_SOURCE_CONTENT}" _CIF_SOURCE_CONTENT)
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${INCLUDE}")
    endif()
    if(${ARGC} EQUAL 3)
      set(CMAKE_C_FLAGS_SAVE ${CMAKE_C_FLAGS})
      string(APPEND CMAKE_C_FLAGS " ${ARGV2}")
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

    try_compile(${VARIABLE}
      SOURCE_FROM_VAR CheckIncludeFile.c _CIF_SOURCE_CONTENT
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${_CIF_LINK_OPTIONS}
      ${_CIF_LINK_LIBRARIES}
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_INCLUDE_FILE_FLAGS}
      "${CHECK_INCLUDE_FILE_C_INCLUDE_DIRS}"
      "${_CIF_LINK_DIRECTORIES}"
      )
    unset(_CIF_LINK_OPTIONS)
    unset(_CIF_LINK_LIBRARIES)
    unset(_CIF_LINK_DIRECTORIES)

    if(${ARGC} EQUAL 3)
      set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS_SAVE})
    endif()

    if(${VARIABLE})
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
      set(${VARIABLE} 1 CACHE INTERNAL "Have include ${INCLUDE}")
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
      set(${VARIABLE} "" CACHE INTERNAL "Have include ${INCLUDE}")
    endif()
  endif()
endmacro()
