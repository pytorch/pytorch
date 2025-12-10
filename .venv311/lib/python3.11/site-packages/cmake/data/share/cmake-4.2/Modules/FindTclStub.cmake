# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindTclStub
-----------

Finds the Tcl Stub Library, which is used for building version-independent Tcl
extensions:

.. code-block:: cmake

  find_package(TclStub [...])

Tcl (Tool Command Language) is a dynamic programming language, and the Tcl Stub
Library provides a mechanism to allow Tcl extensions to be compiled in a way
that they can work across multiple Tcl versions, without requiring
recompilation.

This module is typically used in conjunction with Tcl development projects that
aim to be portable across different Tcl releases.  It first calls the
:module:`FindTCL` module to locate Tcl installation and then attempts to find
the stub libraries corresponding to the located Tcl version.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``TclStub_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether the Tcl Stub Library was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``TCL_STUB_LIBRARY``
  The path to the Tcl stub library.
``TK_STUB_LIBRARY``
  The path to the Tk stub library.
``TTK_STUB_LIBRARY``
  The path to the ttk stub library.

Examples
^^^^^^^^

Finding Tcl Stubs Library:

.. code-block:: cmake

  find_package(TclStub)

See Also
^^^^^^^^

* The :module:`FindTCL` module to find the Tcl installation.
* The :module:`FindTclsh` module to find the Tcl shell command-line executable.

Online references:

* `How to Use the Tcl Stubs Library
  <https://www.tcl-lang.org/doc/howto/stubs.html>`_

* `Practical Programming in Tcl and Tk
  <https://www.oreilly.com/library/view/practical-programming-in/0130385603/>`_
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/FindTCL.cmake)

get_filename_component(TCL_TCLSH_PATH "${TCL_TCLSH}" PATH)
get_filename_component(TCL_TCLSH_PATH_PARENT "${TCL_TCLSH_PATH}" PATH)
string(REGEX REPLACE
  "^.*tclsh([0-9]\\.*[0-9]).*$" "\\1" TCL_TCLSH_VERSION "${TCL_TCLSH}")

get_filename_component(TK_WISH_PATH "${TK_WISH}" PATH)
get_filename_component(TK_WISH_PATH_PARENT "${TK_WISH_PATH}" PATH)
string(REGEX REPLACE
  "^.*wish([0-9]\\.*[0-9]).*$" "\\1" TK_WISH_VERSION "${TK_WISH}")

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

set(TCLTK_POSSIBLE_LIB_PATHS
  "${TCL_INCLUDE_PATH_PARENT}/lib"
  "${TK_INCLUDE_PATH_PARENT}/lib"
  "${TCL_LIBRARY_PATH}"
  "${TK_LIBRARY_PATH}"
  "${TCL_TCLSH_PATH_PARENT}/lib"
  "${TK_WISH_PATH_PARENT}/lib"
)

if(WIN32)
  get_filename_component(
    ActiveTcl_CurrentVersion
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActiveTcl;CurrentVersion]"
    NAME)
  set(TCLTK_POSSIBLE_LIB_PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActiveTcl\\${ActiveTcl_CurrentVersion}]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.6;Root]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.5;Root]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.4;Root]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.3;Root]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.2;Root]/lib"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.0;Root]/lib"
    "$ENV{ProgramFiles}/Tcl/Lib"
    "C:/Program Files/Tcl/lib"
    "C:/Tcl/lib"
    )
endif()

find_library(TCL_STUB_LIBRARY
  NAMES
  tclstub
  tclstub${TK_LIBRARY_VERSION} tclstub${TCL_TCLSH_VERSION} tclstub${TK_WISH_VERSION}
  tclstub87 tclstub8.7
  tclstub86 tclstub8.6
  tclstub85 tclstub8.5
  tclstub84 tclstub8.4
  tclstub83 tclstub8.3
  tclstub82 tclstub8.2
  tclstub80 tclstub8.0
  PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
)

find_library(TK_STUB_LIBRARY
  NAMES
  tkstub
  tkstub${TCL_LIBRARY_VERSION} tkstub${TCL_TCLSH_VERSION} tkstub${TK_WISH_VERSION}
  tkstub87 tkstub8.7
  tkstub86 tkstub8.6
  tkstub85 tkstub8.5
  tkstub84 tkstub8.4
  tkstub83 tkstub8.3
  tkstub82 tkstub8.2
  tkstub80 tkstub8.0
  PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
)

find_library(TTK_STUB_LIBRARY
  NAMES
  ttkstub
  ttkstub${TCL_LIBRARY_VERSION} ttkstub${TCL_TCLSH_VERSION} ttkstub${TK_WISH_VERSION}
  ttkstub88 ttkstub8.8
  ttkstub87 ttkstub8.7
  ttkstub86 ttkstub8.6
  ttkstub85 ttkstub8.5
  PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
)

mark_as_advanced(
  TCL_STUB_LIBRARY
  TK_STUB_LIBRARY
  )

if(TCL_STUB_LIBRARY AND TK_STUB_LIBRARY AND TTK_STUB_LIBRARY)
  set(TclStub_FOUND TRUE)
else()
  set(TclStub_FOUND FALSE)
endif()
