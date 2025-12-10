# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCVS
-------

Finds the Concurrent Versions System (CVS):

.. code-block:: cmake

  find_package(CVS [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``CVS_FOUND``
  Boolean indicating whether the ``cvs`` command-line client was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CVS_EXECUTABLE``
  Path to ``cvs`` command-line client.

Examples
^^^^^^^^

Finding CVS and executing it in a process:

.. code-block:: cmake

  find_package(CVS)
  if(CVS_FOUND)
    execute_process(COMMAND ${CVS_EXECUTABLE} --help)
  endif()
#]=======================================================================]

# CVSNT

get_filename_component(
  CVSNT_TypeLib_Win32
  "[HKEY_CLASSES_ROOT\\TypeLib\\{2BDF7A65-0BFE-4B1A-9205-9AB900C7D0DA}\\1.0\\0\\win32]"
  PATH)

get_filename_component(
  CVSNT_Services_EventMessagePath
  "[HKEY_LOCAL_MACHINE\\SYSTEM\\ControlSet001\\Services\\Eventlog\\Application\\cvsnt;EventMessageFile]"
  PATH)

# WinCVS (in case CVSNT was installed in the same directory)

get_filename_component(
  WinCVS_Folder_Command
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Classes\\Folder\\shell\\wincvs\\command]"
  PATH)

# TortoiseCVS (in case CVSNT was installed in the same directory)

get_filename_component(
  TortoiseCVS_Folder_Command
  "[HKEY_CLASSES_ROOT\\CVS\\shell\\open\\command]"
  PATH)

get_filename_component(
  TortoiseCVS_DefaultIcon
  "[HKEY_CLASSES_ROOT\\CVS\\DefaultIcon]"
  PATH)

find_program(CVS_EXECUTABLE cvs
  ${TortoiseCVS_DefaultIcon}
  ${TortoiseCVS_Folder_Command}
  ${WinCVS_Folder_Command}
  ${CVSNT_Services_EventMessagePath}
  ${CVSNT_TypeLib_Win32}
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\CVS\\Pserver;InstallPath]"
  DOC "CVS command line client"
  )
mark_as_advanced(CVS_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CVS DEFAULT_MSG CVS_EXECUTABLE)
