# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
MacroAddFileDependencies
------------------------

.. deprecated:: 3.14
  Do not use this module in new code.

  Instead use the :command:`set_property` command to append to the
  :prop_sf:`OBJECT_DEPENDS` source file property directly:

  .. code-block:: cmake

    set_property(SOURCE <source> APPEND PROPERTY OBJECT_DEPENDS <files>...)

Load this module in a CMake project with:

.. code-block:: cmake

  include(MacroAddFileDependencies)

Commands
^^^^^^^^

This module provides the following command:

.. command:: macro_add_file_dependencies

  Adds dependencies to a source file:

  .. code-block:: cmake

    macro_add_file_dependencies(<source> <files>...)

  This command adds the given ``<files>`` to the dependencies of file
  ``<source>``.
#]=======================================================================]

macro(MACRO_ADD_FILE_DEPENDENCIES _file)
  set_property(SOURCE "${_file}" APPEND PROPERTY OBJECT_DEPENDS "${ARGN}")
endmacro()
