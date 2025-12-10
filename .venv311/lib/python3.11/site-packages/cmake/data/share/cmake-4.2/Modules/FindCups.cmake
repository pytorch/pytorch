# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCups
--------

Finds the Common UNIX Printing System (CUPS):

.. code-block:: cmake

  find_package(Cups [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Cups::Cups``
  .. versionadded:: 3.15

  Target encapsulating the CUPS usage requirements, available only if CUPS is
  found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Cups_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) CUPS was found.

``Cups_VERSION``
  .. versionadded:: 4.2

  The version of CUPS found.

``CUPS_INCLUDE_DIRS``
  Include directories needed for using CUPS.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CUPS_INCLUDE_DIR``
  The directory containing the CUPS headers.

``CUPS_LIBRARIES``
  Libraries needed to link against to use CUPS.

Hints
^^^^^

This module accepts the following variables:

``CUPS_REQUIRE_IPP_DELETE_ATTRIBUTE``
  Set this variable to ``TRUE`` to require CUPS version which features the
  ``ippDeleteAttribute()`` function (i.e. at least of CUPS ``1.1.19``).

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``CUPS_FOUND``
  .. deprecated:: 4.2
    Use ``Cups_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) CUPS was found.

``CUPS_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Cups_VERSION``.

  The version of CUPS found.

Examples
^^^^^^^^

Finding CUPS and linking it to a project target:

.. code-block:: cmake

  find_package(Cups)
  target_link_libraries(project_target PRIVATE Cups::Cups)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(CUPS_INCLUDE_DIR cups/cups.h )

find_library(CUPS_LIBRARIES NAMES cups )

if (CUPS_INCLUDE_DIR AND CUPS_LIBRARIES AND CUPS_REQUIRE_IPP_DELETE_ATTRIBUTE)
    include(${CMAKE_CURRENT_LIST_DIR}/CheckLibraryExists.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_QUIET ${Cups_FIND_QUIETLY})

    # ippDeleteAttribute is new in cups-1.1.19 (and used by kdeprint)
    check_library_exists(cups ippDeleteAttribute "" CUPS_HAS_IPP_DELETE_ATTRIBUTE)
    cmake_pop_check_state()
endif ()

if (CUPS_INCLUDE_DIR AND EXISTS "${CUPS_INCLUDE_DIR}/cups/cups.h")
    file(STRINGS "${CUPS_INCLUDE_DIR}/cups/cups.h" cups_version_str
         REGEX "^#[\t ]*define[\t ]+CUPS_VERSION_(MAJOR|MINOR|PATCH)[\t ]+[0-9]+$")

    unset(Cups_VERSION)
    foreach(VPART MAJOR MINOR PATCH)
        foreach(VLINE ${cups_version_str})
            if(VLINE MATCHES "^#[\t ]*define[\t ]+CUPS_VERSION_${VPART}[\t ]+([0-9]+)$")
                set(CUPS_VERSION_PART "${CMAKE_MATCH_1}")
                if(Cups_VERSION)
                    string(APPEND Cups_VERSION ".${CUPS_VERSION_PART}")
                else()
                    set(Cups_VERSION "${CUPS_VERSION_PART}")
                endif()
            endif()
        endforeach()
    endforeach()
    set(CUPS_VERSION_STRING ${Cups_VERSION})
endif ()

include(FindPackageHandleStandardArgs)

if (CUPS_REQUIRE_IPP_DELETE_ATTRIBUTE)
    find_package_handle_standard_args(Cups
                                      REQUIRED_VARS CUPS_LIBRARIES CUPS_INCLUDE_DIR CUPS_HAS_IPP_DELETE_ATTRIBUTE
                                      VERSION_VAR Cups_VERSION)
else ()
    find_package_handle_standard_args(Cups
                                      REQUIRED_VARS CUPS_LIBRARIES CUPS_INCLUDE_DIR
                                      VERSION_VAR Cups_VERSION)
endif ()

mark_as_advanced(CUPS_INCLUDE_DIR CUPS_LIBRARIES)

if (Cups_FOUND)
    set(CUPS_INCLUDE_DIRS "${CUPS_INCLUDE_DIR}")
    if (NOT TARGET Cups::Cups)
        add_library(Cups::Cups INTERFACE IMPORTED)
        set_target_properties(Cups::Cups PROPERTIES
            INTERFACE_LINK_LIBRARIES      "${CUPS_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUPS_INCLUDE_DIR}")
    endif ()
endif ()

cmake_policy(POP)
