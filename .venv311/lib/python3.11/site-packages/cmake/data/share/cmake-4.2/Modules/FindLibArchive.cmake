# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLibArchive
--------------

Finds the libarchive library and include directories:

.. code-block:: cmake

  find_package(LibArchive [<version>] [...])

Libarchive is a multi-format archive and compression library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``LibArchive::LibArchive``
  .. versionadded:: 3.17

  A target encapsulating the libarchive usage requirements, available only
  if libarchive is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LibArchive_FOUND``
  Boolean indicating whether (the requested version of) libarchive was found.

``LibArchive_VERSION``
  A 3-component version string (``major.minor.patch``) of libarchive found.

  .. versionadded:: 3.6

    Support for a new libarchive version string format.  Starting from
    libarchive version 3.2, a different preprocessor macro is used in the header
    to define the version.  In CMake 3.5 and earlier, this variable will be set
    only for libarchive versions 3.1 and earlier.  In CMake 3.6 and newer, this
    variable will be set for all libarchive versions.

``LibArchive_INCLUDE_DIRS``
  Include search path for using libarchive.

``LibArchive_LIBRARIES``
  Libraries to link against libarchive.

Examples
^^^^^^^^

Finding libarchive and linking it to a project target:

.. code-block:: cmake

  find_package(LibArchive)
  target_link_libraries(project_target PRIVATE LibArchive::LibArchive)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(LibArchive_INCLUDE_DIR
  NAMES archive.h
  PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\LibArchive;InstallPath]/include"
  DOC "libarchive include directory"
  )

find_library(LibArchive_LIBRARY
  NAMES archive libarchive
  PATHS
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\LibArchive;InstallPath]/lib"
  DOC "libarchive library"
  )

mark_as_advanced(LibArchive_INCLUDE_DIR LibArchive_LIBRARY)

# Extract the version number from the header.
if(LibArchive_INCLUDE_DIR AND EXISTS "${LibArchive_INCLUDE_DIR}/archive.h")
  # The version string appears in one of three known formats in the header:
  #  #define ARCHIVE_LIBRARY_VERSION "libarchive 2.4.12"
  #  #define ARCHIVE_VERSION_STRING "libarchive 2.8.4"
  #  #define ARCHIVE_VERSION_ONLY_STRING "3.2.0"
  # Match any format.
  set(_LibArchive_VERSION_REGEX "^#define[ \t]+ARCHIVE[_A-Z]+VERSION[_A-Z]*[ \t]+\"(libarchive +)?([0-9]+)\\.([0-9]+)\\.([0-9]+)[^\"]*\".*$")
  file(STRINGS "${LibArchive_INCLUDE_DIR}/archive.h" _LibArchive_VERSION_STRING LIMIT_COUNT 1 REGEX "${_LibArchive_VERSION_REGEX}")
  if(_LibArchive_VERSION_STRING)
    string(REGEX REPLACE "${_LibArchive_VERSION_REGEX}" "\\2.\\3.\\4" LibArchive_VERSION "${_LibArchive_VERSION_STRING}")
  endif()
  unset(_LibArchive_VERSION_REGEX)
  unset(_LibArchive_VERSION_STRING)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibArchive
                                  REQUIRED_VARS LibArchive_LIBRARY LibArchive_INCLUDE_DIR
                                  VERSION_VAR LibArchive_VERSION
  )
unset(LIBARCHIVE_FOUND)

if(LibArchive_FOUND)
  set(LibArchive_INCLUDE_DIRS ${LibArchive_INCLUDE_DIR})
  set(LibArchive_LIBRARIES    ${LibArchive_LIBRARY})

  if (NOT TARGET LibArchive::LibArchive)
    add_library(LibArchive::LibArchive UNKNOWN IMPORTED)
    set_target_properties(LibArchive::LibArchive PROPERTIES
      IMPORTED_LOCATION "${LibArchive_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LibArchive_INCLUDE_DIR}")
  endif ()
endif()

cmake_policy(POP)
