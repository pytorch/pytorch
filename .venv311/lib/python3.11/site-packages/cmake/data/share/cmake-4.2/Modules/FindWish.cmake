# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindWish
--------

Finds ``wish``, a simple windowing shell command-line executable:

.. code-block:: cmake

  find_package(Wish [...])

This module is commonly used in conjunction with finding a TCL installation (see
the :module:`FindTCL` module).  It helps determine where the TCL include paths
and libraries are, as well as identifying the name of the TCL library.

If the :variable:`UNIX` variable is defined, the module will prioritize looking
for the Cygwin version of ``wish`` executable.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Wish_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether the ``wish`` executable was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``TK_WISH``
  The path to the ``wish`` executable.

Examples
^^^^^^^^

Finding ``wish``:

.. code-block:: cmake

  find_package(Wish)
  message(STATUS "Found wish at: ${TK_WISH}")
#]=======================================================================]

if(UNIX)
  find_program(TK_WISH cygwish80 )
endif()

get_filename_component(TCL_TCLSH_PATH "${TCL_TCLSH}" PATH)
get_filename_component(TCL_TCLSH_PATH_PARENT "${TCL_TCLSH_PATH}" PATH)
string(REGEX REPLACE
  "^.*tclsh([0-9]\\.*[0-9]).*$" "\\1" TCL_TCLSH_VERSION "${TCL_TCLSH}")

get_filename_component(TCL_INCLUDE_PATH_PARENT "${TCL_INCLUDE_PATH}" PATH)
get_filename_component(TK_INCLUDE_PATH_PARENT "${TK_INCLUDE_PATH}" PATH)

get_filename_component(TCL_LIBRARY_PATH "${TCL_LIBRARY}" PATH)
get_filename_component(TCL_LIBRARY_PATH_PARENT "${TCL_LIBRARY_PATH}" PATH)
string(REGEX REPLACE
  "^.*tcl([0-9]\\.*[0-9]).*$" "\\1" TCL_LIBRARY_VERSION "${TCL_LIBRARY}")

get_filename_component(TK_LIBRARY_PATH "${TK_LIBRARY}" PATH)
get_filename_component(TK_LIBRARY_PATH_PARENT "${TK_LIBRARY_PATH}" PATH)
string(REGEX REPLACE
  "^.*tk([0-9]\\.*[0-9]).*$" "\\1" TK_LIBRARY_VERSION "${TK_LIBRARY}")

set(TCLTK_POSSIBLE_BIN_PATHS
  "${TCL_INCLUDE_PATH_PARENT}/bin"
  "${TK_INCLUDE_PATH_PARENT}/bin"
  "${TCL_LIBRARY_PATH_PARENT}/bin"
  "${TK_LIBRARY_PATH_PARENT}/bin"
  "${TCL_TCLSH_PATH_PARENT}/bin"
  )

if(WIN32)
  get_filename_component(
    ActiveTcl_CurrentVersion
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActiveTcl;CurrentVersion]"
    NAME)
  set(TCLTK_POSSIBLE_BIN_PATHS ${TCLTK_POSSIBLE_BIN_PATHS}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActiveTcl\\${ActiveTcl_CurrentVersion}]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.6;Root]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.5;Root]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.4;Root]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.3;Root]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.2;Root]/bin"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.0;Root]/bin"
    )
endif()

set(TK_WISH_NAMES
  wish
  wish${TCL_LIBRARY_VERSION} wish${TK_LIBRARY_VERSION} wish${TCL_TCLSH_VERSION}
  wish86 wish8.6
  wish85 wish8.5
  wish84 wish8.4
  wish83 wish8.3
  wish82 wish8.2
  wish80 wish8.0
  )

find_program(TK_WISH
  NAMES ${TK_WISH_NAMES}
  HINTS ${TCLTK_POSSIBLE_BIN_PATHS}
  )

mark_as_advanced(TK_WISH)

if(TK_WISH)
  set(Wish_FOUND TRUE)
else()
  set(Wish_FOUND FALSE)
endif()
