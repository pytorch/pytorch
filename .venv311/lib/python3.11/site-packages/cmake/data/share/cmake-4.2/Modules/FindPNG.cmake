# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPNG
-------

Finds libpng, the official reference library for the PNG image format:

.. code-block:: cmake

  find_package(PNG [<version>] [...])

.. note::

  The PNG library depends on the ZLib compression library, which must be found
  for this module to succeed.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``PNG::PNG``
  .. versionadded:: 3.5

  Target encapsulating the libpng library usage requirements, available if
  libpng is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PNG_FOUND``
  Boolean indicating whether the (requested version of) PNG library was found.

``PNG_VERSION``
  .. versionadded:: 4.2

  The version of the PNG library found.

``PNG_INCLUDE_DIRS``
  Directory containing the PNG headers (e.g., ``png.h``).

``PNG_LIBRARIES``
  PNG libraries required for linking.

``PNG_DEFINITIONS``
  Compile definitions for using PNG, if any.  They can be added with
  :command:`target_compile_definitions` command when not using the ``PNG::PNG``
  imported target.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables may also be set for backward compatibility:

``PNG_LIBRARY``
  .. deprecated:: 3.0
    Use the ``PNG::PNG`` imported target.

  Path to the PNG library.

``PNG_INCLUDE_DIR``
  .. deprecated:: 3.0
    Use the ``PNG::PNG`` imported target.

  Directory containing the PNG headers (same as ``PNG_INCLUDE_DIRS``).

``PNG_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``PNG_VERSION``.

  The version of the PNG library found.

Examples
^^^^^^^^

Finding PNG library and using it in a project:

.. code-block:: cmake

  find_package(PNG)
  target_link_libraries(project_target PRIVATE PNG::PNG)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# Default install location on windows when installing from included cmake build
# From FindZLIB.cmake
set(_PNG_x86 "(x86)")
set(_PNG_INCLUDE_SEARCH_NORMAL
  "$ENV{ProgramFiles}/libpng"
  "$ENV{ProgramFiles${_PNG_x86}}/libpng")
set(_PNG_LIB_SEARCH_NORMAL
  "$ENV{ProgramFiles}/libpng/lib"
  "$ENV{ProgramFiles${_PNG_x86}}/libpng/lib")
unset(_PNG_x86)

if(PNG_FIND_QUIETLY)
  set(_FIND_ZLIB_ARG QUIET)
endif()
find_package(ZLIB ${_FIND_ZLIB_ARG})

if(ZLIB_FOUND)
  set(_PNG_VERSION_SUFFIXES 17 16 15 14 12)

  list(APPEND _PNG_INCLUDE_PATH_SUFFIXES include/libpng)
  foreach(v IN LISTS _PNG_VERSION_SUFFIXES)
    list(APPEND _PNG_INCLUDE_PATH_SUFFIXES include/libpng${v})
  endforeach()

  find_path(PNG_PNG_INCLUDE_DIR png.h PATH_SUFFIXES ${_PNG_INCLUDE_PATH_SUFFIXES} PATHS ${_PNG_INCLUDE_SEARCH_NORMAL} )
  mark_as_advanced(PNG_PNG_INCLUDE_DIR)

  list(APPEND PNG_NAMES png libpng)
  unset(PNG_NAMES_DEBUG)
  if (PNG_FIND_VERSION MATCHES "^([0-9]+)\\.([0-9]+)(\\..*)?$")
    set(_PNG_VERSION_SUFFIX_MIN "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    if (PNG_FIND_VERSION_EXACT)
      set(_PNG_VERSION_SUFFIXES ${_PNG_VERSION_SUFFIX_MIN})
    else ()
      string(REGEX REPLACE
          "${_PNG_VERSION_SUFFIX_MIN}.*" "${_PNG_VERSION_SUFFIX_MIN}"
          _PNG_VERSION_SUFFIXES "${_PNG_VERSION_SUFFIXES}")
    endif ()
    unset(_PNG_VERSION_SUFFIX_MIN)
  endif ()
  foreach(v IN LISTS _PNG_VERSION_SUFFIXES)
    list(APPEND PNG_NAMES png${v} libpng${v} libpng${v}_static)
    list(APPEND PNG_NAMES_DEBUG png${v}d libpng${v}d libpng${v}_staticd)
  endforeach()
  unset(_PNG_VERSION_SUFFIXES)
  # For compatibility with versions prior to this multi-config search, honor
  # any PNG_LIBRARY that is already specified and skip the search.
  if(NOT PNG_LIBRARY)
    find_library(PNG_LIBRARY_RELEASE NAMES ${PNG_NAMES} NAMES_PER_DIR PATHS ${_PNG_LIB_SEARCH_NORMAL})
    find_library(PNG_LIBRARY_DEBUG NAMES ${PNG_NAMES_DEBUG} NAMES_PER_DIR PATHS ${_PNG_LIB_SEARCH_NORMAL})
    include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
    select_library_configurations(PNG)
    mark_as_advanced(PNG_LIBRARY_RELEASE PNG_LIBRARY_DEBUG)
  endif()
  unset(PNG_NAMES)
  unset(PNG_NAMES_DEBUG)
  unset(_PNG_INCLUDE_PATH_SUFFIXES)

  # Set by select_library_configurations(), but we want the one from
  # find_package_handle_standard_args() below.
  unset(PNG_FOUND)

  if (PNG_LIBRARY AND PNG_PNG_INCLUDE_DIR)
      # png.h includes zlib.h. Sigh.
      set(PNG_INCLUDE_DIRS ${PNG_PNG_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR} )
      set(PNG_INCLUDE_DIR ${PNG_INCLUDE_DIRS} ) # for backward compatibility
      set(PNG_LIBRARIES ${PNG_LIBRARY} ${ZLIB_LIBRARY})
      if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND
         ("${PNG_LIBRARY}" MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$"))
        list(APPEND PNG_LIBRARIES m)
      endif()

      if (CYGWIN)
        if(BUILD_SHARED_LIBS)
           # No need to define PNG_USE_DLL here, because it's default for Cygwin.
        else()
          set (PNG_DEFINITIONS -DPNG_STATIC)
          set(_PNG_COMPILE_DEFINITIONS PNG_STATIC)
        endif()
      endif ()

      if(NOT TARGET PNG::PNG)
        add_library(PNG::PNG UNKNOWN IMPORTED)
        set_target_properties(PNG::PNG PROPERTIES
          INTERFACE_COMPILE_DEFINITIONS "${_PNG_COMPILE_DEFINITIONS}"
          INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
          INTERFACE_LINK_LIBRARIES ZLIB::ZLIB)
        if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND
           ("${PNG_LIBRARY}" MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$"))
          set_property(TARGET PNG::PNG APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES m)
        endif()

        if(EXISTS "${PNG_LIBRARY}")
          set_target_properties(PNG::PNG PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${PNG_LIBRARY}")
        endif()
        if(EXISTS "${PNG_LIBRARY_RELEASE}")
          set_property(TARGET PNG::PNG APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE)
          set_target_properties(PNG::PNG PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
            IMPORTED_LOCATION_RELEASE "${PNG_LIBRARY_RELEASE}")
        endif()
        if(EXISTS "${PNG_LIBRARY_DEBUG}")
          set_property(TARGET PNG::PNG APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG)
          set_target_properties(PNG::PNG PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${PNG_LIBRARY_DEBUG}")
        endif()
      endif()

      unset(_PNG_COMPILE_DEFINITIONS)
  endif ()

  if (PNG_PNG_INCLUDE_DIR AND EXISTS "${PNG_PNG_INCLUDE_DIR}/png.h")
      file(STRINGS "${PNG_PNG_INCLUDE_DIR}/png.h" png_version_str REGEX "^#define[ \t]+PNG_LIBPNG_VER_STRING[ \t]+\".+\"")

      string(REGEX REPLACE "^#define[ \t]+PNG_LIBPNG_VER_STRING[ \t]+\"([^\"]+)\".*" "\\1" PNG_VERSION "${png_version_str}")
      set(PNG_VERSION_STRING "${PNG_VERSION}")
      unset(png_version_str)
  endif ()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PNG
                                  REQUIRED_VARS PNG_LIBRARY PNG_PNG_INCLUDE_DIR
                                  VERSION_VAR PNG_VERSION)

cmake_policy(POP)
