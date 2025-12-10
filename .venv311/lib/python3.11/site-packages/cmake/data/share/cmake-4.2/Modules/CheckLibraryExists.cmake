# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckLibraryExists
------------------

This module provides a command to check whether a C library exists.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckLibraryExists)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_library_exists

  Checks once whether a specified library exists and a given C function is
  available:

  .. code-block:: cmake

    check_library_exists(<library> <function> <location> <variable>)

  This command attempts to link a test executable that uses the specified
  C ``<function>`` to verify that it is provided by either a system or
  user-provided ``<library>``.

  The arguments are:

  ``<library>``
    The name of the library, a full path to a library file, or an
    :ref:`Imported Target <Imported Targets>`.

  ``<function>``
    The name of a function that should be available in the system or
    user-provided library ``<library>``.

  ``<location>``
    The directory containing the library file.  It is added to the link
    search path during the check.  If this is an empty string, only the
    default library search paths are used.

  ``<variable>``
    The name of the variable in which to store the check result.  This
    variable will be created as an internal cache variable.

  .. note::

    This command is intended for performing basic sanity checks to verify
    that a library provides the expected functionality, or that the correct
    library is being located.  However, it only verifies that a function
    symbol can be linked successfully - it does not ensure that the function
    is declared in library headers, nor can it detect functions that are
    inlined or defined as preprocessor macros.  For more robust detection
    of function availability, prefer using :module:`CheckSymbolExists` or
    :module:`CheckSourceCompiles`.

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

Checking if the ``curl`` library exists in the default paths and has the
``curl_easy_perform()`` function:

.. code-block:: cmake

  include(CheckLibraryExists)
  check_library_exists(curl curl_easy_perform "" HAVE_LIBRARY_CURL)

To check if library exists in specific non-standard location and has a specified
function:

.. code-block:: cmake

  include(CheckLibraryExists)
  check_library_exists(curl curl_easy_perform "/opt/curl/lib" HAVE_LIBRARY_CURL)

Also :ref:`Imported Targets` (for example, from the ``find_package()`` call)
can be used:

.. code-block:: cmake

  find_package(CURL)

  # ...

  if(TARGET CURL::libcurl)
    include(CheckLibraryExists)
    check_library_exists(CURL::libcurl curl_easy_perform "" HAVE_LIBRARY_CURL)
  endif()

See Also
^^^^^^^^

* The :module:`CheckSymbolExists` module to check whether a C symbol exists.
#]=======================================================================]

include_guard(GLOBAL)

macro(CHECK_LIBRARY_EXISTS LIBRARY FUNCTION LOCATION VARIABLE)
  if(NOT DEFINED "${VARIABLE}")
    set(MACRO_CHECK_LIBRARY_EXISTS_DEFINITION
      "-DCHECK_FUNCTION_EXISTS=${FUNCTION} ${CMAKE_REQUIRED_FLAGS}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_START "Looking for ${FUNCTION} in ${LIBRARY}")
    endif()
    set(CHECK_LIBRARY_EXISTS_LINK_OPTIONS)
    if(CMAKE_REQUIRED_LINK_OPTIONS)
      set(CHECK_LIBRARY_EXISTS_LINK_OPTIONS
        LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS})
    endif()
    set(CHECK_LIBRARY_EXISTS_LIBRARIES ${LIBRARY})
    if(CMAKE_REQUIRED_LIBRARIES)
      set(CHECK_LIBRARY_EXISTS_LIBRARIES
        ${CHECK_LIBRARY_EXISTS_LIBRARIES} ${CMAKE_REQUIRED_LIBRARIES})
    endif()
    if(CMAKE_REQUIRED_LINK_DIRECTORIES)
      set(_CLE_LINK_DIRECTORIES
        "-DLINK_DIRECTORIES:STRING=${LOCATION};${CMAKE_REQUIRED_LINK_DIRECTORIES}")
    else()
      set(_CLE_LINK_DIRECTORIES "-DLINK_DIRECTORIES:STRING=${LOCATION}")
    endif()

    if(CMAKE_C_COMPILER_LOADED)
      set(_cle_source CheckFunctionExists.c)
    elseif(CMAKE_CXX_COMPILER_LOADED)
      set(_cle_source CheckFunctionExists.cxx)
    else()
      message(FATAL_ERROR "CHECK_FUNCTION_EXISTS needs either C or CXX language enabled")
    endif()

    try_compile(${VARIABLE}
      SOURCE_FROM_FILE "${_cle_source}" "${CMAKE_ROOT}/Modules/CheckFunctionExists.c"
      COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
      ${CHECK_LIBRARY_EXISTS_LINK_OPTIONS}
      LINK_LIBRARIES ${CHECK_LIBRARY_EXISTS_LIBRARIES}
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS:STRING=${MACRO_CHECK_LIBRARY_EXISTS_DEFINITION}
      "${_CLE_LINK_DIRECTORIES}"
      )
    unset(_cle_source)
    unset(_CLE_LINK_DIRECTORIES)

    if(${VARIABLE})
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_PASS "found")
      endif()
      set(${VARIABLE} 1 CACHE INTERNAL "Have library ${LIBRARY}")
    else()
      if(NOT CMAKE_REQUIRED_QUIET)
        message(CHECK_FAIL "not found")
      endif()
      set(${VARIABLE} "" CACHE INTERNAL "Have library ${LIBRARY}")
    endif()
  endif()
endmacro()
