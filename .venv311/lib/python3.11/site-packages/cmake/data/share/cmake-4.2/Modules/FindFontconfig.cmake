# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindFontconfig
--------------

.. versionadded:: 3.14

Finds Fontconfig, a library for font configuration and customization:

.. code-block:: cmake

  find_package(Fontconfig [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Fontconfig::Fontconfig``
  Target encapsulating the Fontconfig usage requirements, available if
  Fontconfig is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Fontconfig_FOUND``
  Boolean indicating whether (the requested version of) Fontconfig was found.
``Fontconfig_VERSION``
  The version of Fontconfig found.
``Fontconfig_LIBRARIES``
  The libraries to link against to use Fontconfig.
``Fontconfig_INCLUDE_DIRS``
  The include directories containing headers needed to use Fontconfig.
``Fontconfig_COMPILE_OPTIONS``
  Compiler options needed to use Fontconfig.  These should be passed to
  :command:`target_compile_options` when not using the
  ``Fontconfig::Fontconfig`` imported target.

Examples
^^^^^^^^

Finding Fontconfig and linking it to a project target:

.. code-block:: cmake

  find_package(Fontconfig)
  target_link_libraries(project_target PRIVATE Fontconfig::Fontconfig)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# use pkg-config to get the directories and then use these values
# in the find_path() and find_library() calls
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PKG_FONTCONFIG QUIET fontconfig)
endif()
set(Fontconfig_COMPILE_OPTIONS ${PKG_FONTCONFIG_CFLAGS_OTHER})
set(Fontconfig_VERSION ${PKG_FONTCONFIG_VERSION})

find_path( Fontconfig_INCLUDE_DIR
  NAMES
    fontconfig/fontconfig.h
  HINTS
    ${PKG_FONTCONFIG_INCLUDE_DIRS}
    /usr/X11/include
)

find_library( Fontconfig_LIBRARY
  NAMES
    fontconfig
  PATHS
    ${PKG_FONTCONFIG_LIBRARY_DIRS}
)

if (Fontconfig_INCLUDE_DIR AND NOT Fontconfig_VERSION)
  file(STRINGS ${Fontconfig_INCLUDE_DIR}/fontconfig/fontconfig.h _contents REGEX "^#define[ \t]+FC_[A-Z]+[ \t]+[0-9]+$")
  unset(Fontconfig_VERSION)
  foreach(VPART MAJOR MINOR REVISION)
    foreach(VLINE ${_contents})
      if(VLINE MATCHES "^#define[\t ]+FC_${VPART}[\t ]+([0-9]+)$")
        set(Fontconfig_VERSION_PART "${CMAKE_MATCH_1}")
        if(Fontconfig_VERSION)
          string(APPEND Fontconfig_VERSION ".${Fontconfig_VERSION_PART}")
        else()
          set(Fontconfig_VERSION "${Fontconfig_VERSION_PART}")
        endif()
      endif()
    endforeach()
  endforeach()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Fontconfig
  REQUIRED_VARS
    Fontconfig_LIBRARY
    Fontconfig_INCLUDE_DIR
  VERSION_VAR
    Fontconfig_VERSION
)


if(Fontconfig_FOUND AND NOT TARGET Fontconfig::Fontconfig)
  add_library(Fontconfig::Fontconfig UNKNOWN IMPORTED)
  set_target_properties(Fontconfig::Fontconfig PROPERTIES
    IMPORTED_LOCATION "${Fontconfig_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${Fontconfig_COMPILE_OPTIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${Fontconfig_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(Fontconfig_LIBRARY Fontconfig_INCLUDE_DIR)

if(Fontconfig_FOUND)
  set(Fontconfig_LIBRARIES ${Fontconfig_LIBRARY})
  set(Fontconfig_INCLUDE_DIRS ${Fontconfig_INCLUDE_DIR})
endif()

cmake_policy(POP)
