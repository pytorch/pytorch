# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckIncludeFileCXX
-------------------

This module provides a command to check a C++ header file.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckIncludeFileCXX)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_include_file_cxx

  Checks once whether a header file exists and can be included in C++ code:

  .. code-block:: cmake

    check_include_file_cxx(<include> <variable> [<flags>])

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

Checking whether the ``C++23`` header ``<stdfloat>`` exists and storing the
check result in the ``HAVE_STDFLOAT_HEADER`` cache variable:

.. code-block:: cmake

  include(CheckIncludeFileCXX)

  check_include_file_cxx(stdfloat HAVE_STDFLOAT_HEADER)

See Also
^^^^^^^^

* The :module:`CheckIncludeFile` module to check for single C header.
* The :module:`CheckIncludeFiles` module to check for one or more C or
  C++ headers at once.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_INCLUDE_FILE_CXX INCLUDE VARIABLE)
  if(NOT DEFINED "${VARIABLE}" OR "x${${VARIABLE}}" STREQUAL "x${VARIABLE}")
    if(CMAKE_REQUIRED_INCLUDES)
      set(CHECK_INCLUDE_FILE_CXX_INCLUDE_DIRS "-DINCLUDE_DIRECTORIES=${CMAKE_REQUIRED_INCLUDES}")
    else()
      set(CHECK_INCLUDE_FILE_CXX_INCLUDE_DIRS)
    endif()
    set(MACRO_CHECK_INCLUDE_FILE_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CHECK_INCLUDE_FILE_VAR ${INCLUDE})
    file(READ ${CMAKE_ROOT}/Modules/CheckIncludeFile.cxx.in _CIF_SOURCE_CONTENT)
    string(CONFIGURE "${_CIF_SOURCE_CONTENT}" _CIF_SOURCE_CONTENT)
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for C++ include ${INCLUDE}")
    endif()
    if(${ARGC} EQUAL 3)
      set(CMAKE_CXX_FLAGS_SAVE ${CMAKE_CXX_FLAGS})
      string(APPEND CMAKE_CXX_FLAGS " ${ARGV2}")
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
      SOURCE_FROM_VAR CheckIncludeFile.cxx _CIF_SOURCE_CONTENT
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${_CIF_LINK_OPTIONS}
      ${_CIF_LINK_LIBRARIES}
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_INCLUDE_FILE_FLAGS}
      "${CHECK_INCLUDE_FILE_CXX_INCLUDE_DIRS}"
      "${_CIF_LINK_DIRECTORIES}"
      )
    unset(_CIF_LINK_OPTIONS)
    unset(_CIF_LINK_LIBRARIES)
    unset(_CIF_LINK_DIRECTORIES)

    if(${ARGC} EQUAL 3)
      set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_SAVE})
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
