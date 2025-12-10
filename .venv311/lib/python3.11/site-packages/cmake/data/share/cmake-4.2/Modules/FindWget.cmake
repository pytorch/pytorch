# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindWget
--------

This module finds the ``wget`` command-line tool for retrieving content from web
servers:

.. code-block:: cmake

  find_package(Wget [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Wget_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether ``wget`` was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``WGET_EXECUTABLE``
  The full path to the ``wget`` tool.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``WGET_FOUND``
  .. deprecated:: 4.2
    Use ``Wget_FOUND``, which has the same value.

  Boolean indicating whether ``wget`` was found.

Examples
^^^^^^^^

Finding ``wget`` and executing it in a process:

.. code-block:: cmake

  find_package(Wget)
  if(Wget_FOUND)
    execute_process(COMMAND ${WGET_EXECUTABLE} -h)
  endif()

See Also
^^^^^^^^

* The :command:`file(DOWNLOAD)` command to download the given URL to a local
  file.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/FindCygwin.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindMsys.cmake)

find_program(WGET_EXECUTABLE
  wget
  ${CYGWIN_INSTALL_PATH}/bin
  ${MSYS_INSTALL_PATH}/usr/bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Wget DEFAULT_MSG WGET_EXECUTABLE)

mark_as_advanced( WGET_EXECUTABLE )

# WGET option is deprecated.
# use WGET_EXECUTABLE instead.
set (WGET ${WGET_EXECUTABLE})
