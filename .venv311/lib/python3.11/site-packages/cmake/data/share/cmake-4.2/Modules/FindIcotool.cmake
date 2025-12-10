# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindIcotool
-----------

Finds ``icotool``, command-line program for converting and creating Win32 icon
and cursor files:

.. code-block:: cmake

  find_package(Icotool [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Icotool_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) ``icotool`` was
  found.

``Icotool_VERSION``
  .. versionadded:: 4.2

  The version of ``icotool`` found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ICOTOOL_EXECUTABLE``
  The full path to the ``icotool`` tool.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ICOTOOL_FOUND``
  .. deprecated:: 4.2
    Use ``Icotool_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) ``icotool`` was
  found.

``ICOTOOL_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``Icotool_VERSION``, which has the same value.

  The version of ``icotool`` found.

Examples
^^^^^^^^

Finding ``icotool`` and executing it in a process to create ``.ico`` icon from
the source ``.png`` image located in the current source directory:

.. code-block:: cmake

  find_package(Icotool)
  if(Icotool_FOUND)
    execute_process(
      COMMAND
        ${ICOTOOL_EXECUTABLE} -c -o ${CMAKE_CURRENT_BINARY_DIR}/img.ico img.png
    )
  endif()
#]=======================================================================]

find_program(ICOTOOL_EXECUTABLE
  icotool
)

if(ICOTOOL_EXECUTABLE)
  execute_process(
    COMMAND ${ICOTOOL_EXECUTABLE} --version
    OUTPUT_VARIABLE Icotool_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if("${Icotool_VERSION}" MATCHES "^icotool \\([^\\)]*\\) ([0-9\\.]+[^ \n]*)")
    set(Icotool_VERSION "${CMAKE_MATCH_1}")
  else()
    set(Icotool_VERSION "")
  endif()
  set(ICOTOOL_VERSION_STRING "${Icotool_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Icotool
  REQUIRED_VARS ICOTOOL_EXECUTABLE
  VERSION_VAR Icotool_VERSION
)

mark_as_advanced(
  ICOTOOL_EXECUTABLE
)
