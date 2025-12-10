# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGLUT
--------

Finds the OpenGL Utility Toolkit (GLUT) library, which provides a simple API
for creating windows, handling input, and managing events in OpenGL
applications:

.. code-block:: cmake

  find_package(GLUT [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``GLUT::GLUT``
  .. versionadded:: 3.1

  Target encapsulating the GLUT usage requirements, available if GLUT is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GLUT_FOUND``
  Boolean indicating whether GLUT was found.

``GLUT_INCLUDE_DIRS``
  .. versionadded:: 3.23

  Include directories needed to use GLUT.  Starting with CMake 3.23, this
  variable is intended to be used in target usage requirements instead of the
  cache variable ``GLUT_INCLUDE_DIR``, which is intended for finding GLUT.

``GLUT_LIBRARIES``
  List of libraries needed to link against for using GLUT.

Cache Variables
^^^^^^^^^^^^^^^

This module may set the following cache variables depending on platform.
These variables may optionally be set to help this module find the
correct files, but should not be used as result variables:

``GLUT_INCLUDE_DIR``
  The full path to the directory containing ``GL/glut.h`` (without the ``GL/``).

``GLUT_glut_LIBRARY``
  The full path to the ``glut`` library.

``GLUT_Xi_LIBRARY``
  The full path to the dependent ``Xi`` (X Input Device Extension) library on
  some systems.

``GLUT_Xmu_LIBRARY``
  The full path to the dependent ``Xmu`` (X Miscellaneous Utilities) library on
  some systems.

Examples
^^^^^^^^

Finding GLUT and linking it to a project target:

.. code-block:: cmake

  find_package(GLUT)
  target_link_libraries(project_target PRIVATE GLUT::GLUT)
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_GLUT QUIET glut)
  if(NOT PC_GLUT_FOUND)
    pkg_check_modules(PC_GLUT QUIET freeglut)
  endif()
endif()

if(WIN32)
  find_path( GLUT_INCLUDE_DIR NAMES GL/glut.h
    PATHS  ${GLUT_ROOT_PATH}/include
    HINTS ${PC_GLUT_INCLUDE_DIRS})
  mark_as_advanced(GLUT_INCLUDE_DIR)
  find_library( GLUT_glut_LIBRARY_RELEASE NAMES freeglut glut glut32
    PATHS
    ${OPENGL_LIBRARY_DIR}
    ${GLUT_ROOT_PATH}/Release
    HINTS
    ${PC_GLUT_LIBRARY_DIRS}
    )
# N.B. As the pkg-config cannot distinguish between release and debug libraries,
# assume that their hint was the both Debug and Release library.
  find_library( GLUT_glut_LIBRARY_DEBUG NAMES freeglutd
    PATHS
    ${OPENGL_LIBRARY_DIR}
    ${GLUT_ROOT_PATH}/Debug
    HINTS
    ${PC_GLUT_LIBRARY_DIRS}
    )
  mark_as_advanced(GLUT_glut_LIBRARY_RELEASE GLUT_glut_LIBRARY_DEBUG)
  select_library_configurations(GLUT_glut)
elseif(APPLE)
  find_path(GLUT_INCLUDE_DIR glut.h PATHS ${OPENGL_LIBRARY_DIR} HINTS ${PC_GLUT_INCLUDE_DIRS})
  mark_as_advanced(GLUT_INCLUDE_DIR)
  find_library(GLUT_glut_LIBRARY GLUT HINTS ${PC_GLUT_LIBRARY_DIRS} DOC "GLUT library for OSX")
  find_library(GLUT_cocoa_LIBRARY Cocoa DOC "Cocoa framework for OSX")
  mark_as_advanced(GLUT_glut_LIBRARY GLUT_cocoa_LIBRARY)

  if(GLUT_cocoa_LIBRARY AND NOT TARGET GLUT::Cocoa)
    add_library(GLUT::Cocoa UNKNOWN IMPORTED)
    set_target_properties(GLUT::Cocoa PROPERTIES
      IMPORTED_LOCATION "${GLUT_cocoa_LIBRARY}")
  endif()
else()
  if(BEOS)
    set(_GLUT_INC_DIR /boot/develop/headers/os/opengl)
    set(_GLUT_glut_LIB_DIR /boot/develop/lib/x86)
  else()
    find_library( GLUT_Xi_LIBRARY Xi
      /usr/openwin/lib
      )
    mark_as_advanced(GLUT_Xi_LIBRARY)

    find_library( GLUT_Xmu_LIBRARY Xmu
      /usr/openwin/lib
      )
    mark_as_advanced(GLUT_Xmu_LIBRARY)

    if(GLUT_Xi_LIBRARY AND NOT TARGET GLUT::Xi)
      add_library(GLUT::Xi UNKNOWN IMPORTED)
      set_target_properties(GLUT::Xi PROPERTIES
        IMPORTED_LOCATION "${GLUT_Xi_LIBRARY}")
    endif()

    if(GLUT_Xmu_LIBRARY AND NOT TARGET GLUT::Xmu)
      add_library(GLUT::Xmu UNKNOWN IMPORTED)
      set_target_properties(GLUT::Xmu PROPERTIES
        IMPORTED_LOCATION "${GLUT_Xmu_LIBRARY}")
    endif()

  endif ()

  find_path( GLUT_INCLUDE_DIR GL/glut.h
    PATHS
    /usr/include/GL
    /usr/openwin/share/include
    /usr/openwin/include
    /opt/graphics/OpenGL/include
    /opt/graphics/OpenGL/contrib/libglut
    ${_GLUT_INC_DIR}
    HINTS
    ${PC_GLUT_INCLUDE_DIRS}
    )
  mark_as_advanced(GLUT_INCLUDE_DIR)

  find_library( GLUT_glut_LIBRARY glut
    PATHS
    /usr/openwin/lib
    ${_GLUT_glut_LIB_DIR}
    HINTS
    ${PC_GLUT_LIBRARY_DIRS}
    )
  mark_as_advanced(GLUT_glut_LIBRARY)

  unset(_GLUT_INC_DIR)
  unset(_GLUT_glut_LIB_DIR)
endif()

find_package_handle_standard_args(GLUT REQUIRED_VARS GLUT_glut_LIBRARY GLUT_INCLUDE_DIR)

if (GLUT_FOUND)
  # Is -lXi and -lXmu required on all platforms that have it?
  # If not, we need some way to figure out what platform we are on.
  set( GLUT_LIBRARIES
    ${GLUT_glut_LIBRARY}
    )
  set(GLUT_INCLUDE_DIRS
    ${GLUT_INCLUDE_DIR}
    )
  foreach(v GLUT_Xmu_LIBRARY GLUT_Xi_LIBRARY GLUT_cocoa_LIBRARY)
    if(${v})
      list(APPEND GLUT_LIBRARIES ${${v}})
    endif()
  endforeach()

  if(NOT TARGET GLUT::GLUT)
    add_library(GLUT::GLUT UNKNOWN IMPORTED)
    set_target_properties(GLUT::GLUT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GLUT_INCLUDE_DIRS}")
    if(GLUT_glut_LIBRARY_RELEASE)
      set_property(TARGET GLUT::GLUT APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(GLUT::GLUT PROPERTIES
        IMPORTED_LOCATION_RELEASE "${GLUT_glut_LIBRARY_RELEASE}")
    endif()

    if(GLUT_glut_LIBRARY_DEBUG)
      set_property(TARGET GLUT::GLUT APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(GLUT::GLUT PROPERTIES
        IMPORTED_LOCATION_DEBUG "${GLUT_glut_LIBRARY_DEBUG}")
    endif()

    if(NOT GLUT_glut_LIBRARY_RELEASE AND NOT GLUT_glut_LIBRARY_DEBUG)
      set_property(TARGET GLUT::GLUT APPEND PROPERTY
        IMPORTED_LOCATION "${GLUT_glut_LIBRARY}")
    endif()

    if(TARGET GLUT::Xmu)
      set_property(TARGET GLUT::GLUT APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES GLUT::Xmu)
    endif()

    if(TARGET GLUT::Xi)
      set_property(TARGET GLUT::GLUT APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES GLUT::Xi)
    endif()

    if(TARGET GLUT::Cocoa)
      set_property(TARGET GLUT::GLUT APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES GLUT::Cocoa)
    endif()
  endif()
endif()
