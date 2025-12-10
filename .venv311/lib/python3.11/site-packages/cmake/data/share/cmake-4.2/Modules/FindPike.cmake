# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPike
--------

Finds the Pike compiler and interpreter:

.. code-block:: cmake

  find_package(Pike [...])

Pike is interpreted, general purpose, high-level, dynamic programming
language.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Pike_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether Pike was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PIKE_INCLUDE_PATH``
  The directory containing ``program.h``.
``PIKE_EXECUTABLE``
  Full path to the pike binary.

Examples
^^^^^^^^

Finding Pike:

.. code-block:: cmake

  find_package(Pike)
#]=======================================================================]

find_path(PIKE_INCLUDE_PATH program.h
  ${PIKE_POSSIBLE_INCLUDE_PATHS}
  PATH_SUFFIXES include/pike8.0/pike include/pike7.8/pike include/pike7.4/pike)

find_program(PIKE_EXECUTABLE
  NAMES pike8.0 pike 7.8 pike7.4
  )

mark_as_advanced(
  PIKE_EXECUTABLE
  PIKE_INCLUDE_PATH
  )

if(PIKE_EXECUTABLE AND PIKE_INCLUDE_PATH)
  set(Pike_FOUND TRUE)
else()
  set(Pike_FOUND FALSE)
endif()
