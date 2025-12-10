# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindJPEG
--------

Finds the Joint Photographic Experts Group (JPEG) library (``libjpeg``):

.. code-block:: cmake

  find_package(JPEG [<version>] [...])

.. versionchanged:: 3.12
  Debug and Release JPEG library variants are now found separately.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``JPEG::JPEG``
  .. versionadded:: 3.12

  Target encapsulating the JPEG library usage requirements.  It is available
  only when JPEG is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``JPEG_FOUND``
  Boolean indicating whether the (requested version of) JPEG library was
  found.

``JPEG_VERSION``
  .. versionadded:: 3.12

  The version of JPEG library found.

``JPEG_INCLUDE_DIRS``
  Include directories containing headers needed to use JPEG.

``JPEG_LIBRARIES``
  Libraries needed to link to JPEG.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``JPEG_INCLUDE_DIR``
  Directory containing the ``<jpeglib.h>`` and related header files.

``JPEG_LIBRARY_RELEASE``
  .. versionadded:: 3.12

  Path to the release (optimized) variant of the JPEG library.

``JPEG_LIBRARY_DEBUG``
  .. versionadded:: 3.12

  Path to the debug variant of the JPEG library.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``JPEG_LIBRARY``
  .. deprecated:: 3.12
    This variable has been superseded by the ``JPEG_LIBRARY_RELEASE`` and
    ``JPEG_LIBRARY_DEBUG`` variables.

  Path to the JPEG library.

Examples
^^^^^^^^

Finding JPEG library and linking it to a project target:

.. code-block:: cmake

  find_package(JPEG)
  target_link_libraries(project_target PRIVATE JPEG::JPEG)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(JPEG_INCLUDE_DIR jpeglib.h)

set(jpeg_names ${JPEG_NAMES} jpeg jpeg-static libjpeg libjpeg-static)
foreach(name ${jpeg_names})
  list(APPEND jpeg_names_debug "${name}d")
endforeach()

if(NOT JPEG_LIBRARY)
  find_library(JPEG_LIBRARY_RELEASE NAMES ${jpeg_names} NAMES_PER_DIR)
  find_library(JPEG_LIBRARY_DEBUG NAMES ${jpeg_names_debug} NAMES_PER_DIR)
  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(JPEG)
  mark_as_advanced(JPEG_LIBRARY_RELEASE JPEG_LIBRARY_DEBUG)
endif()
unset(jpeg_names)
unset(jpeg_names_debug)

if(JPEG_INCLUDE_DIR)
  file(GLOB _JPEG_CONFIG_HEADERS_FEDORA "${JPEG_INCLUDE_DIR}/jconfig*.h")
  file(GLOB _JPEG_CONFIG_HEADERS_DEBIAN "${JPEG_INCLUDE_DIR}/*/jconfig.h")
  set(_JPEG_CONFIG_HEADERS
    "${JPEG_INCLUDE_DIR}/jpeglib.h"
    ${_JPEG_CONFIG_HEADERS_FEDORA}
    ${_JPEG_CONFIG_HEADERS_DEBIAN})
  foreach (_JPEG_CONFIG_HEADER IN LISTS _JPEG_CONFIG_HEADERS)
    if (NOT EXISTS "${_JPEG_CONFIG_HEADER}")
      continue ()
    endif ()
    file(STRINGS "${_JPEG_CONFIG_HEADER}"
      jpeg_lib_version REGEX "^#define[\t ]+JPEG_LIB_VERSION[\t ]+.*")

    if (NOT jpeg_lib_version)
      continue ()
    endif ()

    string(REGEX REPLACE "^#define[\t ]+JPEG_LIB_VERSION[\t ]+([0-9]+).*"
      "\\1" JPEG_VERSION "${jpeg_lib_version}")
    break ()
  endforeach ()
  unset(jpeg_lib_version)
  unset(_JPEG_CONFIG_HEADER)
  unset(_JPEG_CONFIG_HEADERS)
  unset(_JPEG_CONFIG_HEADERS_FEDORA)
  unset(_JPEG_CONFIG_HEADERS_DEBIAN)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JPEG
  REQUIRED_VARS JPEG_LIBRARY JPEG_INCLUDE_DIR
  VERSION_VAR JPEG_VERSION)

if(JPEG_FOUND)
  set(JPEG_LIBRARIES ${JPEG_LIBRARY})
  set(JPEG_INCLUDE_DIRS "${JPEG_INCLUDE_DIR}")

  if(NOT TARGET JPEG::JPEG)
    add_library(JPEG::JPEG UNKNOWN IMPORTED)
    if(JPEG_INCLUDE_DIRS)
      set_target_properties(JPEG::JPEG PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JPEG_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${JPEG_LIBRARY}")
      set_target_properties(JPEG::JPEG PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${JPEG_LIBRARY}")
    endif()
    if(EXISTS "${JPEG_LIBRARY_RELEASE}")
      set_property(TARGET JPEG::JPEG APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(JPEG::JPEG PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
        IMPORTED_LOCATION_RELEASE "${JPEG_LIBRARY_RELEASE}")
    endif()
    if(EXISTS "${JPEG_LIBRARY_DEBUG}")
      set_property(TARGET JPEG::JPEG APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(JPEG::JPEG PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
        IMPORTED_LOCATION_DEBUG "${JPEG_LIBRARY_DEBUG}")
    endif()
  endif()
endif()

mark_as_advanced(JPEG_LIBRARY JPEG_INCLUDE_DIR)

cmake_policy(POP)
