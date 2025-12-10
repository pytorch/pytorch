# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGLEW
--------

Finds the OpenGL Extension Wrangler Library (GLEW):

.. code-block:: cmake

  find_package(GLEW [<version>] [...])

GLEW is a cross-platform C/C++ library that helps manage OpenGL extensions by
providing efficient run-time mechanisms to query and load OpenGL functionality
beyond the core specification.

.. versionadded:: 3.7
  Debug and Release library variants are found separately.

.. versionadded:: 3.15
  If GLEW is built using its CMake-based build system, it provides a CMake
  package configuration file (``GLEWConfig.cmake``).  This module now takes that
  into account and first attempts to find GLEW in *config mode*.  If the
  configuration file is not available, it falls back to *module mode* and
  searches standard locations.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``GLEW::GLEW``
  .. versionadded:: 3.1

  The main imported target encapsulating the GLEW usage requirements, available
  if GLEW is found.  It maps usage requirements of either ``GLEW::glew`` or
  ``GLEW::glew_s`` target depending on their availability.

``GLEW::glew``
  .. versionadded:: 3.15

  Target encapsulating the usage requirements for a shared GLEW library.  This
  target is available if GLEW is found and static libraries aren't requested via
  the ``GLEW_USE_STATIC_LIBS`` hint variable (see below).

``GLEW::glew_s``
  .. versionadded:: 3.15

  Target encapsulating the usage requirements for a static GLEW library.  This
  target is available if GLEW is found and the ``GLEW_USE_STATIC_LIBS`` hint
  variable is set to boolean true.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GLEW_FOUND``
  Boolean indicating whether (the requested version of) GLEW was found.

``GLEW_VERSION``
  .. versionadded:: 3.15

  The version of GLEW found.

``GLEW_VERSION_MAJOR``
  .. versionadded:: 3.15

  The major version of GLEW found.

``GLEW_VERSION_MINOR``
  .. versionadded:: 3.15

  The minor version of GLEW found.

``GLEW_VERSION_MICRO``
  .. versionadded:: 3.15

  The micro version of GLEW found.

``GLEW_INCLUDE_DIRS``
  Include directories needed to use GLEW library.

``GLEW_LIBRARIES``
  Libraries needed to link against to use GLEW library (shared or static
  depending on configuration).

``GLEW_SHARED_LIBRARIES``
  .. versionadded:: 3.15

  Libraries needed to link against to use shared GLEW library.

``GLEW_STATIC_LIBRARIES``
  .. versionadded:: 3.15

  Libraries needed to link against to use static GLEW library.

Hints
^^^^^

This module accepts the following variables before calling
``find_package(GLEW)`` to influence this module's behavior:

``GLEW_USE_STATIC_LIBS``
  .. versionadded:: 3.15

  Set to boolean true to find static GLEW library and create the
  ``GLEW::glew_s`` imported target for static linkage.

``GLEW_VERBOSE``
  .. versionadded:: 3.15

  Set to boolean true to output a detailed log of this module.  Can be used, for
  example, for debugging.

Examples
^^^^^^^^

Finding GLEW and linking it to a project target:

.. code-block:: cmake

  find_package(GLEW)
  target_link_libraries(project_target PRIVATE GLEW::GLEW)

Using the static GLEW library, if found:

.. code-block:: cmake

  set(GLEW_USE_STATIC_LIBS TRUE)
  find_package(GLEW)
  target_link_libraries(project_target PRIVATE GLEW::GLEW)
#]=======================================================================]

include(FindPackageHandleStandardArgs)
include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

find_package(GLEW CONFIG QUIET)

if(GLEW_FOUND)
  find_package_handle_standard_args(GLEW DEFAULT_MSG GLEW_CONFIG)
  get_target_property(GLEW_INCLUDE_DIRS GLEW::GLEW INTERFACE_INCLUDE_DIRECTORIES)
  set(GLEW_INCLUDE_DIR ${GLEW_INCLUDE_DIRS})
  get_target_property(_GLEW_DEFS GLEW::GLEW INTERFACE_COMPILE_DEFINITIONS)
  if("${_GLEW_DEFS}" MATCHES "GLEW_STATIC")
    get_target_property(GLEW_LIBRARY_DEBUG GLEW::GLEW IMPORTED_LOCATION_DEBUG)
    get_target_property(GLEW_LIBRARY_RELEASE GLEW::GLEW IMPORTED_LOCATION_RELEASE)
  else()
    get_target_property(GLEW_LIBRARY_DEBUG GLEW::GLEW IMPORTED_IMPLIB_DEBUG)
    get_target_property(GLEW_LIBRARY_RELEASE GLEW::GLEW IMPORTED_IMPLIB_RELEASE)
  endif()
  get_target_property(_GLEW_LINK_INTERFACE GLEW::GLEW IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE) # same for debug and release
  list(APPEND GLEW_LIBRARIES ${_GLEW_LINK_INTERFACE})
  list(APPEND GLEW_LIBRARY ${_GLEW_LINK_INTERFACE})
  select_library_configurations(GLEW)
  if("${_GLEW_DEFS}" MATCHES "GLEW_STATIC")
    set(GLEW_STATIC_LIBRARIES ${GLEW_LIBRARIES})
  else()
    set(GLEW_SHARED_LIBRARIES ${GLEW_LIBRARIES})
  endif()
  unset(_GLEW_DEFS)
  unset(_GLEW_LINK_INTERFACE)
  unset(GLEW_LIBRARY)
  unset(GLEW_LIBRARY_DEBUG)
  unset(GLEW_LIBRARY_RELEASE)
  return()
endif()

if(GLEW_VERBOSE)
  message(STATUS "FindGLEW: did not find GLEW CMake config file. Searching for libraries.")
endif()

if(APPLE)
  find_package(OpenGL QUIET)

  if(OpenGL_FOUND)
    if(GLEW_VERBOSE)
      message(STATUS "FindGLEW: Found OpenGL Framework.")
      message(STATUS "FindGLEW: OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
    endif()
  else()
    if(GLEW_VERBOSE)
      message(STATUS "FindGLEW: could not find GLEW library.")
    endif()
    return()
  endif()
endif()


function(__glew_set_find_library_suffix shared_or_static)
  if((UNIX AND NOT APPLE) AND "${shared_or_static}" MATCHES "SHARED")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" PARENT_SCOPE)
  elseif((UNIX AND NOT APPLE) AND "${shared_or_static}" MATCHES "STATIC")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" PARENT_SCOPE)
  elseif(APPLE AND "${shared_or_static}" MATCHES "SHARED")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib;.so" PARENT_SCOPE)
  elseif(APPLE AND "${shared_or_static}" MATCHES "STATIC")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" PARENT_SCOPE)
  elseif(WIN32 AND MINGW AND "${shared_or_static}" MATCHES "SHARED")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a" PARENT_SCOPE)
  elseif(WIN32 AND MINGW AND "${shared_or_static}" MATCHES "STATIC")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" PARENT_SCOPE)
  elseif(WIN32 AND "${shared_or_static}" MATCHES "SHARED")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" PARENT_SCOPE)
  elseif(WIN32 AND "${shared_or_static}" MATCHES "STATIC")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib;.dll.a" PARENT_SCOPE)
  endif()

  if(GLEW_VERBOSE)
    message(STATUS "FindGLEW: CMAKE_FIND_LIBRARY_SUFFIXES for ${shared_or_static}: ${CMAKE_FIND_LIBRARY_SUFFIXES}")
  endif()
endfunction()


if(GLEW_VERBOSE)
  if(DEFINED GLEW_USE_STATIC_LIBS)
    message(STATUS "FindGLEW: GLEW_USE_STATIC_LIBS: ${GLEW_USE_STATIC_LIBS}.")
  else()
    message(STATUS "FindGLEW: GLEW_USE_STATIC_LIBS is undefined. Treated as FALSE.")
  endif()
endif()

find_path(GLEW_INCLUDE_DIR GL/glew.h)
mark_as_advanced(GLEW_INCLUDE_DIR)

set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIR})

if(GLEW_VERBOSE)
  message(STATUS "FindGLEW: GLEW_INCLUDE_DIR: ${GLEW_INCLUDE_DIR}")
  message(STATUS "FindGLEW: GLEW_INCLUDE_DIRS: ${GLEW_INCLUDE_DIRS}")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_arch "x64")
else()
  set(_arch "Win32")
endif()

set(__GLEW_CURRENT_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

__glew_set_find_library_suffix(SHARED)

find_library(GLEW_SHARED_LIBRARY_RELEASE
             NAMES GLEW glew glew32
             NAMES_PER_DIR
             PATH_SUFFIXES lib lib64 libx32 lib/Release/${_arch}
             PATHS ENV GLEW_ROOT)

find_library(GLEW_SHARED_LIBRARY_DEBUG
             NAMES GLEWd glewd glew32d
             NAMES_PER_DIR
             PATH_SUFFIXES lib lib64
             PATHS ENV GLEW_ROOT)


__glew_set_find_library_suffix(STATIC)

find_library(GLEW_STATIC_LIBRARY_RELEASE
             NAMES GLEW glew glew32s
             NAMES_PER_DIR
             PATH_SUFFIXES lib lib64 libx32 lib/Release/${_arch}
             PATHS ENV GLEW_ROOT)

find_library(GLEW_STATIC_LIBRARY_DEBUG
             NAMES GLEWds glewds glew32ds
             NAMES_PER_DIR
             PATH_SUFFIXES lib lib64
             PATHS ENV GLEW_ROOT)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${__GLEW_CURRENT_FIND_LIBRARY_SUFFIXES})
unset(__GLEW_CURRENT_FIND_LIBRARY_SUFFIXES)

select_library_configurations(GLEW_SHARED)
select_library_configurations(GLEW_STATIC)

if(NOT GLEW_USE_STATIC_LIBS)
  set(GLEW_LIBRARIES ${GLEW_SHARED_LIBRARY})
else()
  set(GLEW_LIBRARIES ${GLEW_STATIC_LIBRARY})
endif()


if(GLEW_VERBOSE)
  message(STATUS "FindGLEW: GLEW_SHARED_LIBRARY_RELEASE: ${GLEW_SHARED_LIBRARY_RELEASE}")
  message(STATUS "FindGLEW: GLEW_STATIC_LIBRARY_RELEASE: ${GLEW_STATIC_LIBRARY_RELEASE}")
  message(STATUS "FindGLEW: GLEW_SHARED_LIBRARY_DEBUG: ${GLEW_SHARED_LIBRARY_DEBUG}")
  message(STATUS "FindGLEW: GLEW_STATIC_LIBRARY_DEBUG: ${GLEW_STATIC_LIBRARY_DEBUG}")
  message(STATUS "FindGLEW: GLEW_SHARED_LIBRARY: ${GLEW_SHARED_LIBRARY}")
  message(STATUS "FindGLEW: GLEW_STATIC_LIBRARY: ${GLEW_STATIC_LIBRARY}")
  message(STATUS "FindGLEW: GLEW_LIBRARIES: ${GLEW_LIBRARIES}")
endif()


# Read version from GL/glew.h file
if(EXISTS "${GLEW_INCLUDE_DIR}/GL/glew.h")
  cmake_policy(PUSH)
  cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
  file(STRINGS "${GLEW_INCLUDE_DIR}/GL/glew.h" _contents REGEX "^VERSION_.+ [0-9]+")
  cmake_policy(POP)
  if(_contents)
    string(REGEX REPLACE ".*VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" GLEW_VERSION_MAJOR "${_contents}")
    string(REGEX REPLACE ".*VERSION_MINOR[ \t]+([0-9]+).*" "\\1" GLEW_VERSION_MINOR "${_contents}")
    string(REGEX REPLACE ".*VERSION_MICRO[ \t]+([0-9]+).*" "\\1" GLEW_VERSION_MICRO "${_contents}")
    set(GLEW_VERSION "${GLEW_VERSION_MAJOR}.${GLEW_VERSION_MINOR}.${GLEW_VERSION_MICRO}")
  endif()
endif()

if(GLEW_VERBOSE)
  message(STATUS "FindGLEW: GLEW_VERSION_MAJOR: ${GLEW_VERSION_MAJOR}")
  message(STATUS "FindGLEW: GLEW_VERSION_MINOR: ${GLEW_VERSION_MINOR}")
  message(STATUS "FindGLEW: GLEW_VERSION_MICRO: ${GLEW_VERSION_MICRO}")
  message(STATUS "FindGLEW: GLEW_VERSION: ${GLEW_VERSION}")
endif()

find_package_handle_standard_args(GLEW
                                  REQUIRED_VARS GLEW_INCLUDE_DIRS GLEW_LIBRARIES
                                  VERSION_VAR GLEW_VERSION)

if(NOT GLEW_FOUND)
  if(GLEW_VERBOSE)
    message(STATUS "FindGLEW: could not find GLEW library.")
  endif()
  return()
endif()


if(NOT TARGET GLEW::glew AND NOT GLEW_USE_STATIC_LIBS)
  if(GLEW_VERBOSE)
    message(STATUS "FindGLEW: Creating GLEW::glew imported target.")
  endif()

  add_library(GLEW::glew UNKNOWN IMPORTED)

  set_target_properties(GLEW::glew
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")

  if(APPLE)
    set_target_properties(GLEW::glew
                          PROPERTIES INTERFACE_LINK_LIBRARIES OpenGL::GL)
  endif()

  if(GLEW_SHARED_LIBRARY_RELEASE)
    set_property(TARGET GLEW::glew
                 APPEND
                 PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(GLEW::glew
                          PROPERTIES IMPORTED_LOCATION_RELEASE "${GLEW_SHARED_LIBRARY_RELEASE}")
  endif()

  if(GLEW_SHARED_LIBRARY_DEBUG)
    set_property(TARGET GLEW::glew
                 APPEND
                 PROPERTY IMPORTED_CONFIGURATIONS DEBUG)

    set_target_properties(GLEW::glew
                          PROPERTIES IMPORTED_LOCATION_DEBUG "${GLEW_SHARED_LIBRARY_DEBUG}")
  endif()

elseif(NOT TARGET GLEW::glew_s AND GLEW_USE_STATIC_LIBS)
  if(GLEW_VERBOSE)
    message(STATUS "FindGLEW: Creating GLEW::glew_s imported target.")
  endif()

  add_library(GLEW::glew_s UNKNOWN IMPORTED)

  set_target_properties(GLEW::glew_s
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")

  if(APPLE)
    set_target_properties(GLEW::glew_s
                          PROPERTIES INTERFACE_LINK_LIBRARIES OpenGL::GL)
  endif()

  if(GLEW_STATIC_LIBRARY_RELEASE)
    set_property(TARGET GLEW::glew_s
                 APPEND
                 PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(GLEW::glew_s
                          PROPERTIES IMPORTED_LOCATION_RELEASE "${GLEW_STATIC_LIBRARY_RELEASE}")
  endif()

  if(GLEW_STATIC_LIBRARY_DEBUG)
    set_property(TARGET GLEW::glew_s
                 APPEND
                 PROPERTY IMPORTED_CONFIGURATIONS DEBUG)

    set_target_properties(GLEW::glew_s
                          PROPERTIES IMPORTED_LOCATION_DEBUG "${GLEW_STATIC_LIBRARY_DEBUG}")
  endif()
endif()

if(NOT TARGET GLEW::GLEW)
  if(GLEW_VERBOSE)
    message(STATUS "FindGLEW: Creating GLEW::GLEW imported target.")
  endif()

  add_library(GLEW::GLEW UNKNOWN IMPORTED)

  set_target_properties(GLEW::GLEW
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")

  if(APPLE)
    set_target_properties(GLEW::GLEW
                          PROPERTIES INTERFACE_LINK_LIBRARIES OpenGL::GL)
  endif()

  if(TARGET GLEW::glew)
    if(GLEW_SHARED_LIBRARY_RELEASE)
      set_property(TARGET GLEW::GLEW
                   APPEND
                   PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

      set_target_properties(GLEW::GLEW
                            PROPERTIES IMPORTED_LOCATION_RELEASE "${GLEW_SHARED_LIBRARY_RELEASE}")
    endif()

    if(GLEW_SHARED_LIBRARY_DEBUG)
      set_property(TARGET GLEW::GLEW
                   APPEND
                   PROPERTY IMPORTED_CONFIGURATIONS DEBUG)

      set_target_properties(GLEW::GLEW
                            PROPERTIES IMPORTED_LOCATION_DEBUG "${GLEW_SHARED_LIBRARY_DEBUG}")
    endif()

  elseif(TARGET GLEW::glew_s)
    if(GLEW_STATIC_LIBRARY_RELEASE)
      set_property(TARGET GLEW::GLEW
                   APPEND
                   PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

      set_target_properties(GLEW::GLEW
                            PROPERTIES IMPORTED_LOCATION_RELEASE "${GLEW_STATIC_LIBRARY_RELEASE}")
    endif()

    if(GLEW_STATIC_LIBRARY_DEBUG AND GLEW_USE_STATIC_LIBS)
      set_property(TARGET GLEW::GLEW
                   APPEND
                   PROPERTY IMPORTED_CONFIGURATIONS DEBUG)

      set_target_properties(GLEW::GLEW
                            PROPERTIES IMPORTED_LOCATION_DEBUG "${GLEW_STATIC_LIBRARY_DEBUG}")
    endif()

  elseif(GLEW_VERBOSE)
    message(WARNING "FindGLEW: no `GLEW::glew` or `GLEW::glew_s` target was created. Something went wrong in FindGLEW target creation.")
  endif()
endif()
