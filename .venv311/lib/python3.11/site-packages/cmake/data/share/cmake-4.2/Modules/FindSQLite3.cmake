# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSQLite3
-----------

.. versionadded:: 3.14

Finds the SQLite 3 library:

.. code-block:: cmake

  find_package(SQLite3 [<version>] [...])

SQLite is a small, fast, self-contained, high-reliability, and full-featured
SQL database engine written in C, intended for embedding in applications.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``SQLite::SQLite3``
  Target encapsulating SQLite library usage requirements.  It is available only
  when SQLite is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SQLite3_FOUND``
  Boolean indicating whether the (requested version of) SQLite library was
  found.

``SQLite3_VERSION``
  The version of SQLite library found.

``SQLite3_INCLUDE_DIRS``
  Include directories containing the ``<sqlite3.h>`` and related headers
  needed to use SQLite.

``SQLite3_LIBRARIES``
  Libraries needed to link against to use SQLite.

Examples
^^^^^^^^

Finding the SQLite library and linking it to a project target:

.. code-block:: cmake

  find_package(SQLite3)
  target_link_libraries(project_target PRIVATE SQLite::SQLite3)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_SQLite3 QUIET sqlite3)
endif()

# Look for the necessary header
find_path(SQLite3_INCLUDE_DIR NAMES sqlite3.h
  HINTS
    ${PC_SQLite3_INCLUDE_DIRS}
)
mark_as_advanced(SQLite3_INCLUDE_DIR)

# Look for the necessary library
find_library(SQLite3_LIBRARY NAMES sqlite3 sqlite
  HINTS
    ${PC_SQLite3_LIBRARY_DIRS}
)
mark_as_advanced(SQLite3_LIBRARY)

# Extract version information from the header file
if(SQLite3_INCLUDE_DIR)
    file(STRINGS ${SQLite3_INCLUDE_DIR}/sqlite3.h _ver_line
         REGEX "^#define SQLITE_VERSION  *\"[0-9]+\\.[0-9]+\\.[0-9]+\""
         LIMIT_COUNT 1)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
           SQLite3_VERSION "${_ver_line}")
    unset(_ver_line)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SQLite3
    REQUIRED_VARS SQLite3_LIBRARY SQLite3_INCLUDE_DIR
    VERSION_VAR SQLite3_VERSION)

# Create the imported target
if(SQLite3_FOUND)
    set(SQLite3_INCLUDE_DIRS ${SQLite3_INCLUDE_DIR})
    set(SQLite3_LIBRARIES ${SQLite3_LIBRARY})
    if(NOT TARGET SQLite::SQLite3)
        add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
        set_target_properties(SQLite::SQLite3 PROPERTIES
            IMPORTED_LOCATION             "${SQLite3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SQLite3_INCLUDE_DIR}")
    endif()
endif()

cmake_policy(POP)
