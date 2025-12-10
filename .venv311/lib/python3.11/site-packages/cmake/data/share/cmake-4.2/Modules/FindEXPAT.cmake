# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindEXPAT
---------

Finds the native Expat headers and library:

.. code-block:: cmake

  find_package(EXPAT [<version>] [...])

Expat is a stream-oriented XML parser library written in C.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``EXPAT::EXPAT``
  .. versionadded:: 3.10

  Target encapsulating the Expat library (``expat``) usage requirements.  This
  target is available only if Expat is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``EXPAT_FOUND``
  Boolean indicating whether (the requested version of) Expat was found.

``EXPAT_VERSION``
  .. versionadded:: 4.2

  The version of Expat found.

``EXPAT_INCLUDE_DIRS``
  Include directories containing ``expat.h`` and related headers needed to use
  Expat.

``EXPAT_LIBRARIES``
  Libraries needed to link against to use Expat.

Hints
^^^^^

This module accepts the following variables:

``EXPAT_USE_STATIC_LIBS``
  .. versionadded:: 3.28

  Set to ``TRUE`` to use static libraries.

  .. versionadded:: 3.31

    Implemented on non-Windows platforms.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``EXPAT_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``EXPAT_VERSION``.

  The version of Expat found.

Examples
^^^^^^^^

Finding Expat library and linking it to a project target:

.. code-block:: cmake

  find_package(EXPAT)
  target_link_libraries(project_target PRIVATE EXPAT::EXPAT)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_EXPAT QUIET expat)
endif()

# Look for the header file.
find_path(EXPAT_INCLUDE_DIR NAMES expat.h HINTS ${PC_EXPAT_INCLUDE_DIRS})

set(EXPAT_NAMES expat expatw)
set(EXPAT_NAMES_DEBUG expatd expatwd)

if(WIN32)
  if(EXPAT_USE_STATIC_LIBS)
    list(APPEND EXPAT_NAMES expatMT expatwMT)
    list(APPEND EXPAT_NAMES_DEBUG expatdMT expatwdMT)
  else()
    list(APPEND EXPAT_NAMES expatMT expatMD expatwMT expatwMD)
    list(APPEND EXPAT_NAMES_DEBUG expatdMT expatdMD expatwdMT expatwdMD)
  endif()
endif()

# Allow EXPAT_LIBRARY to be set manually, as the location of the expat library
if(NOT EXPAT_LIBRARY)
  if(DEFINED CMAKE_FIND_LIBRARY_PREFIXES)
    set(_expat_ORIG_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(_expat_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
  endif()

  if(DEFINED CMAKE_FIND_LIBRARY_SUFFIXES)
    set(_expat_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(_expat_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()

  if(WIN32)
    list(APPEND CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
  endif()

  if (EXPAT_USE_STATIC_LIBS)
    if(WIN32)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()
  endif()

  # Look for the library.
  find_library(EXPAT_LIBRARY_RELEASE NAMES ${EXPAT_NAMES} NAMES_PER_DIR HINTS ${PC_EXPAT_LIBRARY_DIRS} PATH_SUFFIXES lib)
  find_library(EXPAT_LIBRARY_DEBUG NAMES ${EXPAT_NAMES_DEBUG} NAMES_PER_DIR HINTS ${PC_EXPAT_LIBRARY_DIRS} PATH_SUFFIXES lib)

  # Restore the original find library ordering
  if(DEFINED _expat_ORIG_CMAKE_FIND_LIBRARY_PREFIXES)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${_expat_ORIG_CMAKE_FIND_LIBRARY_PREFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_PREFIXES)
  endif()

  if(DEFINED _expat_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_expat_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(EXPAT)
endif()

unset(EXPAT_NAMES)
unset(EXPAT_NAMES_DEBUG)

if(EXPAT_INCLUDE_DIR AND EXISTS "${EXPAT_INCLUDE_DIR}/expat.h")
  file(STRINGS "${EXPAT_INCLUDE_DIR}/expat.h" expat_version_str
    REGEX "^#[\t ]*define[\t ]+XML_(MAJOR|MINOR|MICRO)_VERSION[\t ]+[0-9]+$")

  unset(EXPAT_VERSION)
  foreach(VPART MAJOR MINOR MICRO)
    foreach(VLINE ${expat_version_str})
      if(VLINE MATCHES "^#[\t ]*define[\t ]+XML_${VPART}_VERSION[\t ]+([0-9]+)$")
        set(EXPAT_VERSION_PART "${CMAKE_MATCH_1}")
        if(EXPAT_VERSION)
          string(APPEND EXPAT_VERSION ".${EXPAT_VERSION_PART}")
        else()
          set(EXPAT_VERSION "${EXPAT_VERSION_PART}")
        endif()
      endif()
    endforeach()
  endforeach()
  set(EXPAT_VERSION_STRING ${EXPAT_VERSION})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EXPAT
                                  REQUIRED_VARS EXPAT_LIBRARY EXPAT_INCLUDE_DIR
                                  VERSION_VAR EXPAT_VERSION)

# Copy the results to the output variables and target.
if(EXPAT_FOUND)
  set(EXPAT_INCLUDE_DIRS ${EXPAT_INCLUDE_DIR})

  if(NOT EXPAT_LIBRARIES)
    set(EXPAT_LIBRARIES ${EXPAT_LIBRARY})
  endif()

  if(NOT TARGET EXPAT::EXPAT)
    add_library(EXPAT::EXPAT UNKNOWN IMPORTED)
    set_target_properties(EXPAT::EXPAT PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      INTERFACE_INCLUDE_DIRECTORIES "${EXPAT_INCLUDE_DIRS}")

    if(EXPAT_USE_STATIC_LIBS)
      set_property(TARGET EXPAT::EXPAT APPEND PROPERTY
                   INTERFACE_COMPILE_DEFINITIONS "XML_STATIC")
    endif()

    if(EXPAT_LIBRARY_RELEASE)
      set_property(TARGET EXPAT::EXPAT APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(EXPAT::EXPAT PROPERTIES
        IMPORTED_LOCATION_RELEASE "${EXPAT_LIBRARY_RELEASE}")
    endif()

    if(EXPAT_LIBRARY_DEBUG)
      set_property(TARGET EXPAT::EXPAT APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(EXPAT::EXPAT PROPERTIES
        IMPORTED_LOCATION_DEBUG "${EXPAT_LIBRARY_DEBUG}")
    endif()

    if(NOT EXPAT_LIBRARY_RELEASE AND NOT EXPAT_LIBRARY_DEBUG)
      set_property(TARGET EXPAT::EXPAT APPEND PROPERTY
        IMPORTED_LOCATION "${EXPAT_LIBRARY}")
    endif()
  endif()
endif()

mark_as_advanced(EXPAT_INCLUDE_DIR EXPAT_LIBRARY)

cmake_policy(POP)
