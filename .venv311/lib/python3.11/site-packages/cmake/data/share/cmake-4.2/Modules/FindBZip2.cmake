# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindBZip2
---------

Finds the BZip2 data compression library (libbz2):

.. code-block:: cmake

  find_package(BZip2 [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``BZip2::BZip2``
  .. versionadded:: 3.12

  Target encapsulating the usage requirements of BZip2 library.  This target is
  available only when BZip2 is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``BZip2_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) BZip2 library was
  found.

``BZip2_VERSION``
  .. versionadded:: 4.2

  The version of BZip2 found.

``BZIP2_INCLUDE_DIRS``
  .. versionadded:: 3.12

  Include directories needed to use BZip2 library.

``BZIP2_LIBRARIES``
  Libraries needed for linking to use BZip2.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``BZIP2_INCLUDE_DIR``
  The directory containing the BZip2 headers.

``BZIP2_LIBRARY_RELEASE``
  The path to the BZip2 library for release configurations.

``BZIP2_LIBRARY_DEBUG``
  The path to the BZip2 library for debug configurations.

``BZIP2_NEED_PREFIX``
  Boolean indicating whether BZip2 functions are prefixed with ``BZ2_``
  (e.g., ``BZ2_bzCompressInit()``).  Versions of BZip2 prior to 1.0.0 used
  unprefixed function names (e.g., ``bzCompressInit()``).

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``BZIP2_FOUND``
  .. deprecated:: 4.2
    Use ``BZip2_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) BZip2 library was
  found.

``BZIP2_VERSION_STRING``
  .. deprecated:: 3.26
    Superseded by the ``BZIP2_VERSION`` (and ``BZip2_VERSION``).

  The version of BZip2 found.

``BZIP2_VERSION``
  .. versionadded:: 3.26

  .. deprecated:: 4.2
    Superseded by the ``BZip2_VERSION``.

  The version of BZip2 found.

Examples
^^^^^^^^

Finding BZip2 library and linking it to a project target:

.. code-block:: cmake

  find_package(BZip2)
  target_link_libraries(project_target PRIVATE BZip2::BZip2)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

set(_BZIP2_PATHS PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\Bzip2;InstallPath]"
  )

find_path(BZIP2_INCLUDE_DIR bzlib.h ${_BZIP2_PATHS} PATH_SUFFIXES include)

if (NOT BZIP2_LIBRARIES)
    find_library(BZIP2_LIBRARY_RELEASE NAMES bz2 bzip2 libbz2 libbzip2 NAMES_PER_DIR ${_BZIP2_PATHS} PATH_SUFFIXES lib)
    find_library(BZIP2_LIBRARY_DEBUG NAMES bz2d bzip2d libbz2d libbzip2d NAMES_PER_DIR ${_BZIP2_PATHS} PATH_SUFFIXES lib)

    include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
    select_library_configurations(BZIP2)
else ()
    file(TO_CMAKE_PATH "${BZIP2_LIBRARIES}" BZIP2_LIBRARIES)
endif ()

if (BZIP2_INCLUDE_DIR AND EXISTS "${BZIP2_INCLUDE_DIR}/bzlib.h")
    file(STRINGS "${BZIP2_INCLUDE_DIR}/bzlib.h" BZLIB_H REGEX "bzip2/libbzip2 version [0-9]+\\.[^ ]+ of [0-9]+ ")
    string(REGEX REPLACE ".* bzip2/libbzip2 version ([0-9]+\\.[^ ]+) of [0-9]+ .*" "\\1" BZIP2_VERSION_STRING "${BZLIB_H}")
    set(BZIP2_VERSION ${BZIP2_VERSION_STRING})
    set(BZip2_VERSION ${BZIP2_VERSION_STRING})
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BZip2
                                  REQUIRED_VARS BZIP2_LIBRARIES BZIP2_INCLUDE_DIR
                                  VERSION_VAR BZip2_VERSION)

if (BZip2_FOUND)
  set(BZIP2_INCLUDE_DIRS ${BZIP2_INCLUDE_DIR})
  include(${CMAKE_CURRENT_LIST_DIR}/CheckSymbolExists.cmake)
  include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET ${BZip2_FIND_QUIETLY})
  set(CMAKE_REQUIRED_INCLUDES ${BZIP2_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${BZIP2_LIBRARIES})

  # Versions before 1.0.2 required <stdio.h> for the FILE definition.
  set(BZip2_headers "bzlib.h")
  if(BZip2_VERSION VERSION_LESS "1.0.2")
    list(PREPEND BZip2_headers "stdio.h")
  endif()
  check_symbol_exists(BZ2_bzCompressInit "${BZip2_headers}" BZIP2_NEED_PREFIX)
  unset(BZip2_headers)

  cmake_pop_check_state()

  if(NOT TARGET BZip2::BZip2)
    add_library(BZip2::BZip2 UNKNOWN IMPORTED)
    set_target_properties(BZip2::BZip2 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${BZIP2_INCLUDE_DIRS}")

    if(BZIP2_LIBRARY_RELEASE)
      set_property(TARGET BZip2::BZip2 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(BZip2::BZip2 PROPERTIES
        IMPORTED_LOCATION_RELEASE "${BZIP2_LIBRARY_RELEASE}")
    endif()

    if(BZIP2_LIBRARY_DEBUG)
      set_property(TARGET BZip2::BZip2 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(BZip2::BZip2 PROPERTIES
        IMPORTED_LOCATION_DEBUG "${BZIP2_LIBRARY_DEBUG}")
    endif()

    if(NOT BZIP2_LIBRARY_RELEASE AND NOT BZIP2_LIBRARY_DEBUG)
      set_property(TARGET BZip2::BZip2 APPEND PROPERTY
        IMPORTED_LOCATION "${BZIP2_LIBRARY}")
    endif()
  endif()
endif ()

mark_as_advanced(BZIP2_INCLUDE_DIR)

cmake_policy(POP)
