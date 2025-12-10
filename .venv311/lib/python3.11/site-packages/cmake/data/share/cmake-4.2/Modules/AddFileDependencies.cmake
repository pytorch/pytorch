# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
AddFileDependencies
-------------------

.. deprecated:: 3.20
  Do not use this module in new code.

  Instead use the :command:`set_property` command to append to the
  :prop_sf:`OBJECT_DEPENDS` source file property directly:

  .. code-block:: cmake

    set_property(SOURCE <source> APPEND PROPERTY OBJECT_DEPENDS <files>...)

Load this module in a CMake project with:

.. code-block:: cmake

  include(AddFileDependencies)

Commands
^^^^^^^^

This module provides the following command:

.. command:: add_file_dependencies

  Adds dependencies to a source file:

  .. code-block:: cmake

    add_file_dependencies(<source> <files>...)

  This command adds the given ``<files>`` to the dependencies of file
  ``<source>``.
#]=======================================================================]

function(add_file_dependencies _file)
  set_property(SOURCE "${_file}" APPEND PROPERTY OBJECT_DEPENDS "${ARGN}")
endfunction()
