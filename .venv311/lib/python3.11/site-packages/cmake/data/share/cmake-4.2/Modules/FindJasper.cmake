# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindJasper
----------

Finds the JasPer Image Coding Toolkit for handling image data in a variety of
formats, such as the JPEG-2000:

.. code-block:: cmake

  find_package(Jasper [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Jasper::Jasper``
  .. versionadded:: 3.22

  Target encapsulating the JasPer library usage requirements, available only if
  the library is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Jasper_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) JasPer was found.

``Jasper_VERSION``
  .. versionadded:: 4.2

  The version of JasPer found.

``JASPER_INCLUDE_DIRS``
  .. versionadded:: 3.22

  The include directories needed to use the JasPer library.

``JASPER_LIBRARIES``
  The libraries needed to use JasPer.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``JASPER_INCLUDE_DIR``
  The directory containing the ``jasper/jasper.h`` and other headers needed to
  use the JasPer library.

``JASPER_LIBRARY_RELEASE``
  The path to the release (optimized) variant of the JasPer library.

``JASPER_LIBRARY_DEBUG``
  The path to the debug variant of the JasPer library.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``JASPER_FOUND``
  .. deprecated:: 4.2
    Use ``Jasper_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) JasPer was found.

``JASPER_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Jasper_VERSION``.

  The version of JasPer found.

Examples
^^^^^^^^

Finding the JasPer library and linking it to a project target:

.. code-block:: cmake

  find_package(Jasper)
  target_link_libraries(project_target PRIVATE Jasper::Jasper)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(JASPER_INCLUDE_DIR jasper/jasper.h)
mark_as_advanced(JASPER_INCLUDE_DIR)

if(NOT JASPER_LIBRARIES)
  find_package(JPEG QUIET)
  find_library(JASPER_LIBRARY_RELEASE NAMES jasper libjasper)
  find_library(JASPER_LIBRARY_DEBUG NAMES jasperd)
  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(JASPER)
endif()

if(JASPER_INCLUDE_DIR AND EXISTS "${JASPER_INCLUDE_DIR}/jasper/jas_config.h")
  file(STRINGS "${JASPER_INCLUDE_DIR}/jasper/jas_config.h" jasper_version_str REGEX "^#define[\t ]+JAS_VERSION[\t ]+\".*\".*")
  string(REGEX REPLACE "^#define[\t ]+JAS_VERSION[\t ]+\"([^\"]+)\".*" "\\1" Jasper_VERSION "${jasper_version_str}")
  set(JASPER_VERSION_STRING "${Jasper_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jasper
                                  REQUIRED_VARS JASPER_LIBRARIES JASPER_INCLUDE_DIR
                                  VERSION_VAR Jasper_VERSION)

if(Jasper_FOUND)
  set(JASPER_LIBRARIES ${JASPER_LIBRARIES})
  if(JPEG_FOUND)
    list(APPEND JASPER_LIBRARIES ${JPEG_LIBRARIES})
  endif()
  set(JASPER_INCLUDE_DIRS ${JASPER_INCLUDE_DIR})
  if(NOT TARGET Jasper::Jasper)
    add_library(Jasper::Jasper UNKNOWN IMPORTED)
    if(JASPER_INCLUDE_DIRS)
      set_target_properties(Jasper::Jasper PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JASPER_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${JASPER_LIBRARY_RELEASE}")
      set_property(TARGET Jasper::Jasper APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(Jasper::Jasper PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
        IMPORTED_LOCATION_RELEASE "${JASPER_LIBRARY_RELEASE}")
    endif()
    if(EXISTS "${JASPER_LIBRARY_DEBUG}")
      set_property(TARGET Jasper::Jasper APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(Jasper::Jasper PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
        IMPORTED_LOCATION_DEBUG "${JASPER_LIBRARY_DEBUG}")
    endif()
  endif()
endif()

cmake_policy(POP)
