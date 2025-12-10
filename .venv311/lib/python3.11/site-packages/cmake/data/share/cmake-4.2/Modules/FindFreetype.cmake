# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindFreetype
------------

Finds the FreeType font renderer library:

.. code-block:: cmake

  find_package(Freetype [<version>] [...])

.. versionadded:: 3.7
  Debug and Release (optimized) library variants are found separately.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Freetype::Freetype``
  .. versionadded:: 3.10

  Target encapsulating the Freetype library usage requirements, available if
  Freetype is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Freetype_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) Freetype was found.

``Freetype_VERSION``
  .. versionadded:: 4.2

  The version of Freetype found.

``FREETYPE_INCLUDE_DIRS``
  Include directories containing headers needed to use Freetype.  This is the
  concatenation of ``FREETYPE_INCLUDE_DIR_ft2build`` and
  ``FREETYPE_INCLUDE_DIR_freetype2`` variables.

``FREETYPE_LIBRARIES``
  Libraries needed to link against for using Freetype.

.. versionadded:: 3.7
  Debug and Release library variants are found separately.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``FREETYPE_INCLUDE_DIR_ft2build``
  The directory containing the main Freetype API configuration header.

``FREETYPE_INCLUDE_DIR_freetype2``
  The directory containing Freetype public headers.

Hints
^^^^^

This module accepts the following variables:

``FREETYPE_DIR``
  The user may set this environment variable to the root directory of a Freetype
  installation to find Freetype in non-standard locations.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``FREETYPE_FOUND``
  .. deprecated:: 4.2
    Use ``Freetype_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) Freetype was found.

``FREETYPE_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Freetype_VERSION``.

  The version of Freetype found.

Examples
^^^^^^^^

Finding Freetype and linking it to a project target:

.. code-block:: cmake

  find_package(Freetype)
  target_link_libraries(project_target PRIVATE Freetype::Freetype)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# Created by Eric Wing.
# Modifications by Alexander Neundorf.
# This file has been renamed to "FindFreetype.cmake" instead of the correct
# "FindFreeType.cmake" in order to be compatible with the one from KDE4, Alex.

# Ugh, FreeType seems to use some #include trickery which
# makes this harder than it should be. It looks like they
# put ft2build.h in a common/easier-to-find location which
# then contains a #include to a more specific header in a
# more specific location (#include <freetype/config/ftheader.h>).
# Then from there, they need to set a bunch of #define's
# so you can do something like:
# #include FT_FREETYPE_H
# Unfortunately, using CMake's mechanisms like include_directories()
# wants explicit full paths and this trickery doesn't work too well.
# I'm going to attempt to cut out the middleman and hope
# everything still works.

set(FREETYPE_FIND_ARGS
  HINTS
    ENV FREETYPE_DIR
  PATHS
    ENV GTKMM_BASEPATH
    [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]
)

find_path(
  FREETYPE_INCLUDE_DIR_ft2build
  ft2build.h
  ${FREETYPE_FIND_ARGS}
  PATH_SUFFIXES
    include/freetype2
    include
    freetype2
)

find_path(
  FREETYPE_INCLUDE_DIR_freetype2
  NAMES
    freetype/config/ftheader.h
    config/ftheader.h
  ${FREETYPE_FIND_ARGS}
  PATH_SUFFIXES
    include/freetype2
    include
    freetype2
)

if(NOT FREETYPE_LIBRARY)
  find_library(FREETYPE_LIBRARY_RELEASE
    NAMES
      freetype
      libfreetype
      freetype219
    ${FREETYPE_FIND_ARGS}
    PATH_SUFFIXES
      lib
  )
  find_library(FREETYPE_LIBRARY_DEBUG
    NAMES
      freetyped
      libfreetyped
      freetype219d
    ${FREETYPE_FIND_ARGS}
    PATH_SUFFIXES
      lib
  )
  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(FREETYPE)
else()
  # on Windows, ensure paths are in canonical format (forward slahes):
  file(TO_CMAKE_PATH "${FREETYPE_LIBRARY}" FREETYPE_LIBRARY)
endif()

unset(FREETYPE_FIND_ARGS)

# set the user variables
if(FREETYPE_INCLUDE_DIR_ft2build AND FREETYPE_INCLUDE_DIR_freetype2)
  set(FREETYPE_INCLUDE_DIRS "${FREETYPE_INCLUDE_DIR_ft2build};${FREETYPE_INCLUDE_DIR_freetype2}")
  list(REMOVE_DUPLICATES FREETYPE_INCLUDE_DIRS)
endif()
set(FREETYPE_LIBRARIES "${FREETYPE_LIBRARY}")

if(EXISTS "${FREETYPE_INCLUDE_DIR_freetype2}/freetype/freetype.h")
  set(FREETYPE_H "${FREETYPE_INCLUDE_DIR_freetype2}/freetype/freetype.h")
elseif(EXISTS "${FREETYPE_INCLUDE_DIR_freetype2}/freetype.h")
  set(FREETYPE_H "${FREETYPE_INCLUDE_DIR_freetype2}/freetype.h")
endif()

if(FREETYPE_INCLUDE_DIR_freetype2 AND FREETYPE_H)
  file(STRINGS "${FREETYPE_H}" freetype_version_str
       REGEX "^#[\t ]*define[\t ]+FREETYPE_(MAJOR|MINOR|PATCH)[\t ]+[0-9]+$")

  unset(Freetype_VERSION)
  foreach(VPART MAJOR MINOR PATCH)
    foreach(VLINE ${freetype_version_str})
      if(VLINE MATCHES "^#[\t ]*define[\t ]+FREETYPE_${VPART}[\t ]+([0-9]+)$")
        set(FREETYPE_VERSION_PART "${CMAKE_MATCH_1}")
        if(Freetype_VERSION)
          string(APPEND Freetype_VERSION ".${FREETYPE_VERSION_PART}")
        else()
          set(Freetype_VERSION "${FREETYPE_VERSION_PART}")
        endif()
        unset(FREETYPE_VERSION_PART)
      endif()
    endforeach()
  endforeach()
  set(FREETYPE_VERSION_STRING ${Freetype_VERSION})
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  Freetype
  REQUIRED_VARS
    FREETYPE_LIBRARY
    FREETYPE_INCLUDE_DIRS
  VERSION_VAR
    Freetype_VERSION
)

mark_as_advanced(
  FREETYPE_INCLUDE_DIR_freetype2
  FREETYPE_INCLUDE_DIR_ft2build
)

if(Freetype_FOUND)
  if(NOT TARGET Freetype::Freetype)
    add_library(Freetype::Freetype UNKNOWN IMPORTED)
    set_target_properties(Freetype::Freetype PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FREETYPE_INCLUDE_DIRS}")

    if(FREETYPE_LIBRARY_RELEASE)
      set_property(TARGET Freetype::Freetype APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(Freetype::Freetype PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
        IMPORTED_LOCATION_RELEASE "${FREETYPE_LIBRARY_RELEASE}")
    endif()

    if(FREETYPE_LIBRARY_DEBUG)
      set_property(TARGET Freetype::Freetype APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(Freetype::Freetype PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
        IMPORTED_LOCATION_DEBUG "${FREETYPE_LIBRARY_DEBUG}")
    endif()

    if(NOT FREETYPE_LIBRARY_RELEASE AND NOT FREETYPE_LIBRARY_DEBUG)
      set_target_properties(Freetype::Freetype PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${FREETYPE_LIBRARY}")
    endif()
  endif()
endif()

cmake_policy(POP)
