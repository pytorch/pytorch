# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindIconv
---------

.. versionadded:: 3.11

Finds the ``iconv()`` POSIX.1 functions on the system:

.. code-block:: cmake

  find_package(Iconv [<version>] [...])

Iconv functions might be provided in the standard C library or externally
in the form of an additional library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Iconv::Iconv``
  Target encapsulating the iconv usage requirements, available only if iconv is
  found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Iconv_FOUND``
  Boolean indicating whether the (requested version of) iconv support was
  found.

``Iconv_VERSION``
  .. versionadded:: 3.21

  The version of iconv found (x.y).

  .. note::
    Some libiconv implementations don't embed the version in their header files.
    In this case the variables ``Iconv_VERSION*`` will be empty.

``Iconv_VERSION_MAJOR``
  .. versionadded:: 3.21

  The major version of iconv.

``Iconv_VERSION_MINOR``
  .. versionadded:: 3.21

  The minor version of iconv.

``Iconv_INCLUDE_DIRS``
  The include directories containing the iconv headers.

``Iconv_LIBRARIES``
  The iconv libraries to be linked.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Iconv_IS_BUILT_IN``
  A boolean variable indicating whether iconv support is stemming from the C
  standard library or not.  Even if the C library provides ``iconv()``, the
  presence of an external ``libiconv`` implementation might lead to this being
  false.

``Iconv_INCLUDE_DIR``
  The directory containing the iconv headers.

``Iconv_LIBRARY``
  The iconv library (if not implicitly given in the C library).

.. note::
  On POSIX platforms, iconv might be part of the C library and the cache
  variables ``Iconv_INCLUDE_DIR`` and ``Iconv_LIBRARY`` might be empty.

Examples
^^^^^^^^

Finding iconv and linking it to a project target:

.. code-block:: cmake

  find_package(Iconv)
  target_link_libraries(project_target PRIVATE Iconv::Iconv)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
if(CMAKE_C_COMPILER_LOADED)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckCSourceCompiles.cmake)
elseif(CMAKE_CXX_COMPILER_LOADED)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckCXXSourceCompiles.cmake)
else()
  # If neither C nor CXX are loaded, implicit iconv makes no sense.
  set(Iconv_IS_BUILT_IN FALSE)
endif()

# iconv can only be provided in libc on a POSIX system.
# If any cache variable is already set, we'll skip this test.
if(NOT DEFINED Iconv_IS_BUILT_IN)
  if(UNIX AND NOT DEFINED Iconv_INCLUDE_DIR AND NOT DEFINED Iconv_LIBRARY)
    cmake_push_check_state(RESET)
    # We always suppress the message here: Otherwise on supported systems
    # not having iconv in their C library (e.g. those using libiconv)
    # would always display a confusing "Looking for iconv - not found" message
    set(CMAKE_FIND_QUIETLY TRUE)
    # The following code will not work, but it's sufficient to see if it compiles.
    # Note: libiconv will define the iconv functions as macros, so CheckSymbolExists
    # will not yield correct results.
    set(Iconv_IMPLICIT_TEST_CODE
      "
      #include <stddef.h>
      #include <iconv.h>
      int main(void) {
        char *a, *b;
        size_t i, j;
        iconv_t ic;
        ic = iconv_open(\"to\", \"from\");
        iconv(ic, &a, &i, &b, &j);
        iconv_close(ic);
      }
      "
    )
    if(CMAKE_C_COMPILER_LOADED)
      check_c_source_compiles("${Iconv_IMPLICIT_TEST_CODE}" Iconv_IS_BUILT_IN)
    else()
      check_cxx_source_compiles("${Iconv_IMPLICIT_TEST_CODE}" Iconv_IS_BUILT_IN)
    endif()
    cmake_pop_check_state()
  else()
    set(Iconv_IS_BUILT_IN FALSE)
  endif()
endif()

set(_Iconv_REQUIRED_VARS)
if(Iconv_IS_BUILT_IN)
  set(_Iconv_REQUIRED_VARS _Iconv_IS_BUILT_IN_MSG)
  set(_Iconv_IS_BUILT_IN_MSG "built in to C library")
else()
  set(_Iconv_REQUIRED_VARS Iconv_LIBRARY Iconv_INCLUDE_DIR)

  find_path(Iconv_INCLUDE_DIR
    NAMES "iconv.h"
    PATH_SUFFIXES
      gnu-libiconv # GNU libiconv on Alpine Linux has header in a subdirectory.
    DOC "iconv include directory")
  set(Iconv_LIBRARY_NAMES "iconv" "libiconv")
  mark_as_advanced(Iconv_INCLUDE_DIR)

  find_library(Iconv_LIBRARY
    NAMES iconv libiconv
    NAMES_PER_DIR
    DOC "iconv library (if not in the C library)")
  mark_as_advanced(Iconv_LIBRARY)
endif()

# NOTE: glibc's iconv.h does not define _LIBICONV_VERSION
if(Iconv_INCLUDE_DIR AND EXISTS "${Iconv_INCLUDE_DIR}/iconv.h")
  file(STRINGS ${Iconv_INCLUDE_DIR}/iconv.h Iconv_VERSION_DEFINE REGEX "_LIBICONV_VERSION (.*)")

  if(Iconv_VERSION_DEFINE MATCHES "(0x[A-Fa-f0-9]+)")
    set(Iconv_VERSION_NUMBER "${CMAKE_MATCH_1}")
    # encoding -> version number: (major<<8) + minor
    math(EXPR Iconv_VERSION_MAJOR "${Iconv_VERSION_NUMBER} >> 8" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR Iconv_VERSION_MINOR "${Iconv_VERSION_NUMBER} - (${Iconv_VERSION_MAJOR} << 8)" OUTPUT_FORMAT HEXADECIMAL)

    math(EXPR Iconv_VERSION_MAJOR "${Iconv_VERSION_MAJOR}" OUTPUT_FORMAT DECIMAL)
    math(EXPR Iconv_VERSION_MINOR "${Iconv_VERSION_MINOR}" OUTPUT_FORMAT DECIMAL)
    set(Iconv_VERSION "${Iconv_VERSION_MAJOR}.${Iconv_VERSION_MINOR}")
  endif()

  unset(Iconv_VERSION_DEFINE)
  unset(Iconv_VERSION_NUMBER)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Iconv
                                  REQUIRED_VARS ${_Iconv_REQUIRED_VARS}
                                  VERSION_VAR Iconv_VERSION)

if(Iconv_FOUND)
  if(Iconv_IS_BUILT_IN)
    set(Iconv_INCLUDE_DIRS "")
    set(Iconv_LIBRARIES "")
  else()
    set(Iconv_INCLUDE_DIRS "${Iconv_INCLUDE_DIR}")
    set(Iconv_LIBRARIES "${Iconv_LIBRARY}")
  endif()
  if(NOT TARGET Iconv::Iconv)
    add_library(Iconv::Iconv INTERFACE IMPORTED)
    set_property(TARGET Iconv::Iconv PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Iconv_INCLUDE_DIRS}")
    set_property(TARGET Iconv::Iconv PROPERTY INTERFACE_LINK_LIBRARIES "${Iconv_LIBRARIES}")
  endif()
endif()

cmake_policy(POP)
