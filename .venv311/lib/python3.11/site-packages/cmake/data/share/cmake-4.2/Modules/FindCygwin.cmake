# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCygwin
----------

Finds Cygwin, a POSIX-compatible environment that runs natively on Microsoft
Windows:

.. code-block:: cmake

  find_package(Cygwin [...])

.. note::

  This module is primarily intended for use in other :ref:`Find Modules` to help
  locate programs when using the ``find_*()`` commands, such as
  :command:`find_program`.  In most cases, direct use of those commands is
  sufficient.  Use this module only if a specific program is known to be
  installed via Cygwin and is usable from Windows.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Cygwin_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether Cygwin was found.

``CYGWIN_INSTALL_PATH``
  The path to the Cygwin root installation directory.

Examples
^^^^^^^^

Finding the Cygwin installation and using its path in a custom find module:

.. code-block:: cmake
  :caption: ``FindFoo.cmake``

  find_package(Cygwin)
  find_program(Foo_EXECUTABLE NAMES foo PATHS ${CYGWIN_INSTALL_PATH}/bin)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Foo REQUIRED_VARS Foo_EXECUTABLE)

See Also
^^^^^^^^

* The :module:`FindMsys` module to find MSYS path in a similar way.
#]=======================================================================]

if (WIN32)
  if(CYGWIN_INSTALL_PATH)
    set(CYGWIN_BAT "${CYGWIN_INSTALL_PATH}/cygwin.bat")
  endif()

  find_program(CYGWIN_BAT
    NAMES cygwin.bat
    PATHS
      "C:/Cygwin"
      "C:/Cygwin64"
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Cygwin\\setup;rootdir]"
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Cygnus Solutions\\Cygwin\\mounts v2\\/;native]"
  )
  get_filename_component(CYGWIN_INSTALL_PATH "${CYGWIN_BAT}" DIRECTORY)
  mark_as_advanced(CYGWIN_BAT)

endif ()

if(CYGWIN_BAT AND CYGWIN_INSTALL_PATH)
  set(Cygwin_FOUND TRUE)
else()
  set(Cygwin_FOUND FALSE)
endif()
