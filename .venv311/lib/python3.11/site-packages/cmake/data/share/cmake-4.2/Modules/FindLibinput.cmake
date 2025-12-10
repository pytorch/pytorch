# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLibinput
------------

.. versionadded:: 3.14

Finds the libinput library which handles input devices in Wayland compositors
and provides a generic X.Org input driver:

.. code-block:: cmake

  find_package(Libinput [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Libinput::Libinput``
  Target encapsulating the libinput library usage requirements, available only
  if library is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Libinput_FOUND``
  Boolean indicating whether the (requested version of) libinput library was
  found.
``Libinput_VERSION``
  The version of the libinput found.
``Libinput_LIBRARIES``
  The libraries to link against to use the libinput library.
``Libinput_INCLUDE_DIRS``
  The include directories containing headers needed to use the libinput library.
``Libinput_COMPILE_OPTIONS``
  Compile options needed to use the libinput library.  These can be passed to
  the :command:`target_compile_options` command, when not using the
  ``Libinput::Libinput`` imported target.

Examples
^^^^^^^^

Finding the libinput library and linking it to a project target:

.. code-block:: cmake

  find_package(Libinput)
  target_link_libraries(project_target PRIVATE Libinput::Libinput)
#]=======================================================================]


# Use pkg-config to get the directories and then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PKG_Libinput QUIET libinput)
endif()

set(Libinput_COMPILE_OPTIONS ${PKG_Libinput_CFLAGS_OTHER})
set(Libinput_VERSION ${PKG_Libinput_VERSION})

find_path(Libinput_INCLUDE_DIR
  NAMES
    libinput.h
  HINTS
    ${PKG_Libinput_INCLUDE_DIRS}
)
find_library(Libinput_LIBRARY
  NAMES
    input
  HINTS
    ${PKG_Libinput_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libinput
  REQUIRED_VARS
    Libinput_LIBRARY
    Libinput_INCLUDE_DIR
  VERSION_VAR
    Libinput_VERSION
)

if(Libinput_FOUND AND NOT TARGET Libinput::Libinput)
  add_library(Libinput::Libinput UNKNOWN IMPORTED)
  set_target_properties(Libinput::Libinput PROPERTIES
    IMPORTED_LOCATION "${Libinput_LIBRARY}"
    INTERFACE_COMPILE_OPTIONS "${Libinput_COMPILE_OPTIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${Libinput_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(Libinput_LIBRARY Libinput_INCLUDE_DIR)

if(Libinput_FOUND)
  set(Libinput_LIBRARIES ${Libinput_LIBRARY})
  set(Libinput_INCLUDE_DIRS ${Libinput_INCLUDE_DIR})
endif()
