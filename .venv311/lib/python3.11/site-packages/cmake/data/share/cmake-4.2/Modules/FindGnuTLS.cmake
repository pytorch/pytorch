# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGnuTLS
----------

Finds the GNU Transport Layer Security library (GnuTLS):

.. code-block:: cmake

  find_package(GnuTLS [<version>] [...])

The GnuTLS package includes the main libraries (libgnutls and libdane), as
well as the optional gnutls-openssl compatibility extra library.  They are
all distributed as part of the same release.  This module checks for the
presence of the main libgnutls library and provides usage requirements for
integrating GnuTLS into CMake projects.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``GnuTLS::GnuTLS``
  .. versionadded:: 3.16

  Target encapsulating the GnuTLS usage requirements, available if GnuTLS is
  found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GnuTLS_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) GnuTLS was found.

``GnuTLS_VERSION``
  .. versionadded:: 4.2

  The version of GnuTLS found.

``GNUTLS_INCLUDE_DIRS``
  Include directories needed to use GnuTLS.

``GNUTLS_LIBRARIES``
  Libraries needed to link against to use GnuTLS.

``GNUTLS_DEFINITIONS``
  Compiler options required for using GnuTLS.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GNUTLS_INCLUDE_DIR``
  The directory containing the ``gnutls/gnutls.h`` header file.

``GNUTLS_LIBRARY``
  The path to the GnuTLS library.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``GNUTLS_FOUND``
  .. deprecated:: 4.2
    Use ``GnuTLS_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) GnuTLS was found.

``GNUTLS_VERSION_STRING``
  .. deprecated:: 3.16
    Use the ``GnuTLS_VERSION``, which has the same value.

``GNUTLS_VERSION``
  .. versionadded:: 3.16
  .. deprecated:: 4.2
    Use the ``GnuTLS_VERSION``, which has the same value.

Examples
^^^^^^^^

Finding GnuTLS and linking it to a project target:

.. code-block:: cmake

  find_package(GnuTLS)
  target_link_libraries(project_target PRIVATE GnuTLS::GnuTLS)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if (GNUTLS_INCLUDE_DIR AND GNUTLS_LIBRARY)
  # in cache already
  set(gnutls_FIND_QUIETLY TRUE)
endif ()

if (NOT WIN32)
  # try using pkg-config to get the directories and then use these values
  # in the find_path() and find_library() calls
  # also fills in GNUTLS_DEFINITIONS, although that isn't normally useful
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_GNUTLS QUIET gnutls)
  endif()
  set(GNUTLS_DEFINITIONS ${PC_GNUTLS_CFLAGS_OTHER})
endif ()

find_path(GNUTLS_INCLUDE_DIR gnutls/gnutls.h
  HINTS
    ${PC_GNUTLS_INCLUDEDIR}
    ${PC_GNUTLS_INCLUDE_DIRS}
  )

find_library(GNUTLS_LIBRARY NAMES gnutls libgnutls
  HINTS
    ${PC_GNUTLS_LIBDIR}
    ${PC_GNUTLS_LIBRARY_DIRS}
  )

mark_as_advanced(GNUTLS_INCLUDE_DIR GNUTLS_LIBRARY)

if(GNUTLS_INCLUDE_DIR AND EXISTS "${GNUTLS_INCLUDE_DIR}/gnutls/gnutls.h")
  file(
    STRINGS
    "${GNUTLS_INCLUDE_DIR}/gnutls/gnutls.h"
    gnutls_version
    # GnuTLS versions prior to 2.7.2 defined LIBGNUTLS_VERSION instead of the
    # current GNUTLS_VERSION.
    REGEX "^#define[\t ]+(LIB)?GNUTLS_VERSION[\t ]+\".*\""
  )

  string(
    REGEX REPLACE
    "^.*GNUTLS_VERSION[\t ]+\"([^\"]*)\".*$"
    "\\1"
    GnuTLS_VERSION
    "${gnutls_version}"
  )
  unset(gnutls_version)

  # Fallback to version defined by pkg-config if not successful.
  if(
    NOT GnuTLS_VERSION
    AND PC_GNUTLS_VERSION
    AND GNUTLS_INCLUDE_DIR IN_LIST PC_GNUTLS_INCLUDE_DIRS
  )
    set(GnuTLS_VERSION "${PC_GNUTLS_VERSION}")
  endif()

  # For backward compatibility.
  set(GNUTLS_VERSION "${GnuTLS_VERSION}")
  set(GNUTLS_VERSION_STRING "${GnuTLS_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GnuTLS
                                  REQUIRED_VARS GNUTLS_LIBRARY GNUTLS_INCLUDE_DIR
                                  VERSION_VAR GnuTLS_VERSION)

if(GnuTLS_FOUND)
  set(GNUTLS_LIBRARIES    ${GNUTLS_LIBRARY})
  set(GNUTLS_INCLUDE_DIRS ${GNUTLS_INCLUDE_DIR})

  if(NOT TARGET GnuTLS::GnuTLS)
    add_library(GnuTLS::GnuTLS UNKNOWN IMPORTED)
    set_target_properties(GnuTLS::GnuTLS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GNUTLS_INCLUDE_DIRS}"
      INTERFACE_COMPILE_DEFINITIONS "${GNUTLS_DEFINITIONS}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${GNUTLS_LIBRARIES}")
  endif()
endif()

cmake_policy(POP)
