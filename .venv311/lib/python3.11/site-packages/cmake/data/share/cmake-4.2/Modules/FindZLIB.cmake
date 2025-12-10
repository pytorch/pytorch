# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindZLIB
--------

Finds the native zlib data compression library:

.. code-block:: cmake

  find_package(ZLIB [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``ZLIB::ZLIB``
  .. versionadded:: 3.1

  Target that encapsulates the zlib usage requirements.  It is available only
  when zlib is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ZLIB_FOUND``
  Boolean indicating whether (the requested version of) zlib was found.

``ZLIB_VERSION``
  .. versionadded:: 3.26

  The version of zlib found.

``ZLIB_INCLUDE_DIRS``
  Include directories containing ``zlib.h`` and other headers needed to use
  zlib.

``ZLIB_LIBRARIES``
  List of libraries needed to link to zlib.

  .. versionchanged:: 3.4
    Debug and Release library variants can be now found separately.

Hints
^^^^^

This module accepts the following variables:

``ZLIB_ROOT``
  A user may set this variable to a zlib installation root to help locate zlib
  in custom installation paths.

``ZLIB_USE_STATIC_LIBS``
  .. versionadded:: 3.24

  Set this variable to ``ON`` before calling ``find_package(ZLIB)`` to look for
  static libraries.  Default is ``OFF``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ZLIB_VERSION_MAJOR``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The major version of zlib.

``ZLIB_VERSION_MINOR``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The minor version of zlib.

``ZLIB_VERSION_PATCH``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The patch version of zlib.

``ZLIB_VERSION_TWEAK``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The tweak version of zlib.

``ZLIB_VERSION_STRING``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The version of zlib found (x.y.z).

``ZLIB_MAJOR_VERSION``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The major version of zlib.

``ZLIB_MINOR_VERSION``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The minor version of zlib.

``ZLIB_PATCH_VERSION``
  .. deprecated:: 3.26
    Superseded by ``ZLIB_VERSION``.

  The patch version of zlib.

Examples
^^^^^^^^

Finding zlib and linking it to a project target:

.. code-block:: cmake

  find_package(ZLIB)
  target_link_libraries(project_target PRIVATE ZLIB::ZLIB)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(ZLIB_FIND_COMPONENTS AND NOT ZLIB_FIND_QUIETLY)
  message(AUTHOR_WARNING
    "ZLIB does not provide any COMPONENTS.  Calling\n"
    "  find_package(ZLIB COMPONENTS ...)\n"
    "will always fail."
    )
endif()

set(_ZLIB_SEARCHES)

# Search ZLIB_ROOT first if it is set.
if(ZLIB_ROOT)
  set(_ZLIB_SEARCH_ROOT PATHS ${ZLIB_ROOT} NO_DEFAULT_PATH)
  list(APPEND _ZLIB_SEARCHES _ZLIB_SEARCH_ROOT)
endif()

# Normal search.
set(_ZLIB_x86 "(x86)")
set(_ZLIB_SEARCH_NORMAL
    PATHS "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\Zlib;InstallPath]"
          "$ENV{ProgramFiles}/zlib"
          "$ENV{ProgramFiles${_ZLIB_x86}}/zlib")
unset(_ZLIB_x86)
list(APPEND _ZLIB_SEARCHES _ZLIB_SEARCH_NORMAL)

if(ZLIB_USE_STATIC_LIBS)
  set(ZLIB_NAMES zs zlibstatic zlibstat zlib z)
  set(ZLIB_NAMES_DEBUG zsd zlibstaticd zlibstatd zlibd zd)
else()
  set(ZLIB_NAMES z zlib zdll zlib1 zlibstatic zlibwapi zlibvc zlibstat)
  set(ZLIB_NAMES_DEBUG zd zlibd zdlld zlibd1 zlib1d zlibstaticd zlibwapid zlibvcd zlibstatd)
endif()

# Try each search configuration.
foreach(search ${_ZLIB_SEARCHES})
  find_path(ZLIB_INCLUDE_DIR NAMES zlib.h ${${search}} PATH_SUFFIXES include)
endforeach()

# Allow ZLIB_LIBRARY to be set manually, as the location of the zlib library
if(NOT ZLIB_LIBRARY)
  if(DEFINED CMAKE_FIND_LIBRARY_PREFIXES)
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
  endif()
  if(DEFINED CMAKE_FIND_LIBRARY_SUFFIXES)
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()
  # Prefix/suffix of the win32/Makefile.gcc build
  if(WIN32)
    list(APPEND CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".dll.a")
  endif()
  # Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
  if(ZLIB_USE_STATIC_LIBS)
    if(WIN32)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()
  endif()

  foreach(search ${_ZLIB_SEARCHES})
    find_library(ZLIB_LIBRARY_RELEASE NAMES ${ZLIB_NAMES} NAMES_PER_DIR ${${search}} PATH_SUFFIXES lib)
    find_library(ZLIB_LIBRARY_DEBUG NAMES ${ZLIB_NAMES_DEBUG} NAMES_PER_DIR ${${search}} PATH_SUFFIXES lib)
  endforeach()

  # Restore the original find library ordering
  if(DEFINED _zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_zlib_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()
  if(DEFINED _zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${_zlib_ORIG_CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_PREFIXES)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(ZLIB)
endif()

unset(ZLIB_NAMES)
unset(ZLIB_NAMES_DEBUG)

mark_as_advanced(ZLIB_INCLUDE_DIR)

if(ZLIB_INCLUDE_DIR AND EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
  file(STRINGS "${ZLIB_INCLUDE_DIR}/zlib.h" ZLIB_H REGEX "^#define ZLIB_VERSION \"[^\"]*\"$")
  if(ZLIB_H MATCHES "ZLIB_VERSION \"(([0-9]+)\\.([0-9]+)(\\.([0-9]+)(\\.([0-9]+))?)?)")
    set(ZLIB_VERSION_STRING "${CMAKE_MATCH_1}")
    set(ZLIB_VERSION_MAJOR "${CMAKE_MATCH_2}")
    set(ZLIB_VERSION_MINOR "${CMAKE_MATCH_3}")
    set(ZLIB_VERSION_PATCH "${CMAKE_MATCH_5}")
    set(ZLIB_VERSION_TWEAK "${CMAKE_MATCH_7}")
  else()
    set(ZLIB_VERSION_STRING "")
    set(ZLIB_VERSION_MAJOR "")
    set(ZLIB_VERSION_MINOR "")
    set(ZLIB_VERSION_PATCH "")
    set(ZLIB_VERSION_TWEAK "")
  endif()
  set(ZLIB_MAJOR_VERSION "${ZLIB_VERSION_MAJOR}")
  set(ZLIB_MINOR_VERSION "${ZLIB_VERSION_MINOR}")
  set(ZLIB_PATCH_VERSION "${ZLIB_VERSION_PATCH}")
  set(ZLIB_VERSION "${ZLIB_VERSION_STRING}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZLIB REQUIRED_VARS ZLIB_LIBRARY ZLIB_INCLUDE_DIR
                                       VERSION_VAR ZLIB_VERSION
                                       HANDLE_COMPONENTS)

if(ZLIB_FOUND)
    set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})

    if(NOT ZLIB_LIBRARIES)
      set(ZLIB_LIBRARIES ${ZLIB_LIBRARY})
    endif()

    if(NOT TARGET ZLIB::ZLIB)
      add_library(ZLIB::ZLIB UNKNOWN IMPORTED)
      set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}")

      if(ZLIB_LIBRARY_RELEASE)
        set_property(TARGET ZLIB::ZLIB APPEND PROPERTY
          IMPORTED_CONFIGURATIONS RELEASE)
        set_target_properties(ZLIB::ZLIB PROPERTIES
          IMPORTED_LOCATION_RELEASE "${ZLIB_LIBRARY_RELEASE}")
      endif()

      if(ZLIB_LIBRARY_DEBUG)
        set_property(TARGET ZLIB::ZLIB APPEND PROPERTY
          IMPORTED_CONFIGURATIONS DEBUG)
        set_target_properties(ZLIB::ZLIB PROPERTIES
          IMPORTED_LOCATION_DEBUG "${ZLIB_LIBRARY_DEBUG}")
      endif()

      if(NOT ZLIB_LIBRARY_RELEASE AND NOT ZLIB_LIBRARY_DEBUG)
        set_property(TARGET ZLIB::ZLIB APPEND PROPERTY
          IMPORTED_LOCATION "${ZLIB_LIBRARY}")
      endif()
    endif()
endif()

cmake_policy(POP)
