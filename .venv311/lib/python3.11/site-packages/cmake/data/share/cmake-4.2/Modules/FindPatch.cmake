# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPatch
---------

.. versionadded:: 3.10

Finds the ``patch`` command-line executable for applying diff patches to
original files:

.. code-block:: cmake

  find_package(Patch [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Patch::patch``
  Target encapsulating the ``patch`` command-line executable, available only if
  ``patch`` is found.

  .. versionchanged:: 4.0
    This imported target is defined only when :prop_gbl:`CMAKE_ROLE` is
    ``PROJECT``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Patch_FOUND``
  Boolean indicating whether the ``patch`` command-line executable was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Patch_EXECUTABLE``
  The path to the ``patch`` command-line executable.

Examples
^^^^^^^^

Finding ``patch`` command and executing it in a process:

.. code-block:: cmake

  find_package(Patch)
  if(Patch_FOUND)
    execute_process(
      COMMAND ${Patch_EXECUTABLE} -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/src.patch
    )
  endif()

The imported target can be used, for example, inside the
:command:`add_custom_command` command, which patches the given file when some
build rule depends on its output:

.. code-block:: cmake

  find_package(Patch)
  if(TARGET Patch::patch)
    # Executed when some build rule depends on the src.c file.
    add_custom_command(
      OUTPUT src.c
      COMMAND Patch::patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/src.patch
      # ...
    )
  endif()
#]=======================================================================]

set(_doc "Patch command line executable")
set(_patch_path )

if(CMAKE_HOST_WIN32)
  set(_patch_path
    "$ENV{LOCALAPPDATA}/Programs/Git/bin"
    "$ENV{LOCALAPPDATA}/Programs/Git/usr/bin"
    "$ENV{APPDATA}/Programs/Git/bin"
    "$ENV{APPDATA}/Programs/Git/usr/bin"
    )
endif()

# First search the PATH
find_program(Patch_EXECUTABLE
  NAMES patch
  PATHS ${_patch_path}
  DOC ${_doc}
  )

if(CMAKE_HOST_WIN32)
  # Now look for installations in Git/ directories under typical installation
  # prefixes on Windows.
  find_program(Patch_EXECUTABLE
    NAMES patch
    PATH_SUFFIXES Git/usr/bin Git/bin GnuWin32/bin
    DOC ${_doc}
    )
endif()

mark_as_advanced(Patch_EXECUTABLE)

get_property(_patch_role GLOBAL PROPERTY CMAKE_ROLE)

if(
  _patch_role STREQUAL "PROJECT"
  AND Patch_EXECUTABLE
  AND NOT TARGET Patch::patch
)
  add_executable(Patch::patch IMPORTED)
  set_property(TARGET Patch::patch PROPERTY IMPORTED_LOCATION ${Patch_EXECUTABLE})
endif()

unset(_patch_path)
unset(_patch_role)
unset(_doc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Patch
                                  REQUIRED_VARS Patch_EXECUTABLE)
