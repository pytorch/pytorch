# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindIntl
--------

.. versionadded:: 3.2

Finds internationalization support that includes message translation functions
such as ``gettext()``:

.. code-block:: cmake

  find_package(Intl [<version>] [...])

These functions originate from the GNU ``libintl`` library, which is part
of the GNU gettext utilities, but may also be provided by the standard C
library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Intl::Intl``
  .. versionadded:: 3.20

  Target encapsulating the Intl usage requirements, available if Intl is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Intl_FOUND``
  Boolean indicating whether (the requested version of) Intl was found.

``Intl_VERSION``
  .. versionadded:: 3.21

  The version of the found Intl implementation or library, in the format
  ``x.y.z``.

  .. note::
    Some Intl implementations don't embed the version in their header files.
    In this case the variables ``Intl_VERSION*`` will be empty.

``Intl_VERSION_MAJOR``
  .. versionadded:: 3.21

  The major version of Intl found.

``Intl_VERSION_MINOR``
  .. versionadded:: 3.21

  The minor version of Intl found.

``Intl_VERSION_PATCH``
  .. versionadded:: 3.21

  The patch version of Intl found.

``Intl_INCLUDE_DIRS``
  Include directories containing headers needed to use Intl.

``Intl_LIBRARIES``
  The libraries needed to link against to use Intl.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Intl_IS_BUILT_IN``
  .. versionadded:: 3.20

  Boolean indicating whether the found Intl functionality is provided by the
  standard C library rather than a separate ``libintl`` library.

``Intl_INCLUDE_DIR``
  The directory containing the ``libintl.h`` header file.

``Intl_LIBRARY``
  The path to the Intl library (if any).

.. note::
  On some platforms, such as Linux with GNU libc, the gettext functions are
  present in the C standard library and libintl is not required.  The
  ``Intl_LIBRARY`` and ``Intl_INCLUDE_DIR`` will be empty in this case.

Examples
^^^^^^^^

Finding the Intl support and linking the imported target for use in a project:

.. code-block:: cmake

  find_package(Intl)
  target_link_libraries(app PRIVATE Intl::Intl)

See Also
^^^^^^^^

* The :module:`FindGettext` module to find and use the GNU gettext tools
  (``msgmerge``, ``msgfmt``, etc.).
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
if(CMAKE_C_COMPILER_LOADED)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckCSourceCompiles.cmake)
elseif(CMAKE_CXX_COMPILER_LOADED)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckCXXSourceCompiles.cmake)
else()
  # If neither C nor CXX are loaded, implicit intl makes no sense.
  set(Intl_IS_BUILT_IN FALSE)
endif()

# Check if Intl is built in to the C library.
if(NOT DEFINED Intl_IS_BUILT_IN)
  if(NOT DEFINED Intl_INCLUDE_DIR AND NOT DEFINED Intl_LIBRARY)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_QUIET TRUE)
    set(Intl_IMPLICIT_TEST_CODE [[
#include <libintl.h>
int main(void) {
  gettext("");
  dgettext("", "");
  dcgettext("", "", 0);
  return 0;
}
]])
    if(CMAKE_C_COMPILER_LOADED)
      check_c_source_compiles("${Intl_IMPLICIT_TEST_CODE}" Intl_IS_BUILT_IN)
    else()
      check_cxx_source_compiles("${Intl_IMPLICIT_TEST_CODE}" Intl_IS_BUILT_IN)
    endif()
    cmake_pop_check_state()
  else()
    set(Intl_IS_BUILT_IN FALSE)
  endif()
endif()

set(_Intl_REQUIRED_VARS)
if(Intl_IS_BUILT_IN)
  set(_Intl_REQUIRED_VARS _Intl_IS_BUILT_IN_MSG)
  set(_Intl_IS_BUILT_IN_MSG "built in to C library")
else()
  set(_Intl_REQUIRED_VARS Intl_LIBRARY Intl_INCLUDE_DIR)

  find_path(Intl_INCLUDE_DIR
            NAMES "libintl.h"
            DOC "libintl include directory")
  mark_as_advanced(Intl_INCLUDE_DIR)

  find_library(Intl_LIBRARY
    NAMES "intl" "libintl"
    NAMES_PER_DIR
    DOC "libintl libraries (if not in the C library)")
  mark_as_advanced(Intl_LIBRARY)
endif()

# NOTE: glibc's libintl.h does not define LIBINTL_VERSION
if(Intl_INCLUDE_DIR AND EXISTS "${Intl_INCLUDE_DIR}/libintl.h")
  file(STRINGS ${Intl_INCLUDE_DIR}/libintl.h Intl_VERSION_DEFINE REGEX "LIBINTL_VERSION (.*)")

  if(Intl_VERSION_DEFINE MATCHES "(0x[A-Fa-f0-9]+)")
    set(Intl_VERSION_NUMBER "${CMAKE_MATCH_1}")
    # encoding -> version number: (major<<16) + (minor<<8) + patch
    math(EXPR Intl_VERSION_MAJOR "${Intl_VERSION_NUMBER} >> 16" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR Intl_VERSION_MINOR "(${Intl_VERSION_NUMBER} - (${Intl_VERSION_MAJOR} << 16)) >> 8" OUTPUT_FORMAT HEXADECIMAL)
    math(EXPR Intl_VERSION_PATCH "${Intl_VERSION_NUMBER} - ((${Intl_VERSION_MAJOR} << 16) + (${Intl_VERSION_MINOR} << 8))" OUTPUT_FORMAT HEXADECIMAL)

    math(EXPR Intl_VERSION_MAJOR "${Intl_VERSION_MAJOR}" OUTPUT_FORMAT DECIMAL)
    math(EXPR Intl_VERSION_MINOR "${Intl_VERSION_MINOR}" OUTPUT_FORMAT DECIMAL)
    math(EXPR Intl_VERSION_PATCH "${Intl_VERSION_PATCH}" OUTPUT_FORMAT DECIMAL)
    set(Intl_VERSION "${Intl_VERSION_MAJOR}.${Intl_VERSION_MINOR}.${Intl_VERSION_PATCH}")
  endif()

  unset(Intl_VERSION_DEFINE)
  unset(Intl_VERSION_NUMBER)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Intl
                                  REQUIRED_VARS ${_Intl_REQUIRED_VARS}
                                  VERSION_VAR Intl_VERSION
                                  FAIL_MESSAGE "Failed to find Gettext libintl")
unset(_Intl_REQUIRED_VARS)
unset(_Intl_IS_BUILT_IN_MSG)

if(Intl_FOUND)
  if(Intl_IS_BUILT_IN)
    set(Intl_INCLUDE_DIRS "")
    set(Intl_LIBRARIES "")
  else()
    set(Intl_INCLUDE_DIRS "${Intl_INCLUDE_DIR}")
    set(Intl_LIBRARIES "${Intl_LIBRARY}")
  endif()
  if(NOT TARGET Intl::Intl)
    add_library(Intl::Intl INTERFACE IMPORTED)
    set_target_properties(Intl::Intl PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Intl_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${Intl_LIBRARIES}")
  endif()
endif()

cmake_policy(POP)
