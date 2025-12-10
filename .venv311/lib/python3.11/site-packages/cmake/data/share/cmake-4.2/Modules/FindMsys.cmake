# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMsys
--------

.. versionadded:: 3.21

Finds MSYS, a POSIX-compatible environment that runs natively on Microsoft
Windows:

.. code-block:: cmake

  find_package(Msys [...])

.. note::

  This module is primarily intended for use in other :ref:`Find Modules` to help
  locate programs when using the ``find_*()`` commands, such as
  :command:`find_program`.  In most cases, direct use of those commands is
  sufficient.  Use this module only if a specific program is known to be
  installed via MSYS and is usable from Windows.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Msys_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether MSYS was found.

``MSYS_INSTALL_PATH``
  The path to the MSYS root installation directory.

Examples
^^^^^^^^

Finding the MSYS installation and using its path in a custom find module:

.. code-block:: cmake
  :caption: ``FindFoo.cmake``

  find_package(Msys)
  find_program(Foo_EXECUTABLE NAMES foo PATHS ${MSYS_INSTALL_PATH}/usr/bin)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Foo REQUIRED_VARS Foo_EXECUTABLE)

See Also
^^^^^^^^

* The :module:`FindCygwin` module to find Cygwin path in a similar way.
#]=======================================================================]

if (WIN32)
  if(MSYS_INSTALL_PATH)
    set(MSYS_CMD "${MSYS_INSTALL_PATH}/msys2_shell.cmd")
  endif()

  find_program(MSYS_CMD
    NAMES msys2_shell.cmd
    PATHS
      # Typical install path for MSYS2 (https://repo.msys2.org/distrib/msys2-i686-latest.sfx.exe)
      "C:/msys32"
      # Typical install path for MSYS2 (https://repo.msys2.org/distrib/msys2-x86_64-latest.sfx.exe)
      "C:/msys64"
      # Git for Windows (https://gitforwindows.org/)
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GitForWindows;InstallPath]"
  )
  get_filename_component(MSYS_INSTALL_PATH "${MSYS_CMD}" DIRECTORY)
  mark_as_advanced(MSYS_CMD)

endif ()

if(MSYS_CMD AND MSYS_INSTALL_PATH)
  set(Msys_FOUND TRUE)
else()
  set(Msys_FOUND FALSE)
endif()
