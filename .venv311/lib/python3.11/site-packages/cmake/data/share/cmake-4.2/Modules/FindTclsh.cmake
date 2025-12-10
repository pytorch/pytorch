# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindTclsh
---------

Finds the Tcl shell command-line executable (``tclsh``), which includes the Tcl
(Tool Command Language) interpreter:

.. code-block:: cmake

  find_package(Tclsh [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Tclsh_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) ``tclsh`` executable
  was found.

``Tclsh_VERSION``
  .. versionadded:: 4.2

  The version of ``tclsh`` found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``TCL_TCLSH``
  The path to the ``tclsh`` executable.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``TCLSH_FOUND``
  .. deprecated:: 4.2
    Use ``Tclsh_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) ``tclsh`` executable
  was found.

``TCLSH_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``Tclsh_VERSION``, which has the same value.

  The version of ``tclsh`` found.

Examples
^^^^^^^^

In the following example, this module is used to find the ``tclsh``
command-line executable, which is then executed in a process to evaluate
TCL code from the script file located in the project source directory:

.. code-block:: cmake

  find_package(Tclsh)
  if(Tclsh_FOUND)
    execute_process(COMMAND ${TCL_TCLSH} example-script.tcl)
  endif()

See Also
^^^^^^^^

* The :module:`FindTCL` module to find the Tcl installation.
* The :module:`FindTclStub` module to find the Tcl Stubs Library.
#]=======================================================================]

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

set(TCLTK_POSSIBLE_BIN_PATHS
  "${TCL_INCLUDE_PATH_PARENT}/bin"
  "${TK_INCLUDE_PATH_PARENT}/bin"
  "${TCL_LIBRARY_PATH_PARENT}/bin"
  "${TK_LIBRARY_PATH_PARENT}/bin"
  "${TK_WISH_PATH_PARENT}/bin"
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

set(TCL_TCLSH_NAMES
  tclsh
  tclsh${TCL_LIBRARY_VERSION} tclsh${TK_LIBRARY_VERSION} tclsh${TK_WISH_VERSION}
  tclsh87 tclsh8.7
  tclsh86 tclsh8.6
  tclsh85 tclsh8.5
  tclsh84 tclsh8.4
  tclsh83 tclsh8.3
  tclsh82 tclsh8.2
  tclsh80 tclsh8.0
  )

find_program(TCL_TCLSH
  NAMES ${TCL_TCLSH_NAMES}
  HINTS ${TCLTK_POSSIBLE_BIN_PATHS}
  )

if(TCL_TCLSH)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E echo puts "\$tcl_version"
                  COMMAND "${TCL_TCLSH}"
                  OUTPUT_VARIABLE Tclsh_VERSION
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(TCLSH_VERSION_STRING "${Tclsh_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
if (CMAKE_FIND_PACKAGE_NAME STREQUAL "TCL" OR
    CMAKE_FIND_PACKAGE_NAME STREQUAL "TclStub")
  # FindTCL include()'s this module. It's an old pattern, but rather than
  # trying to suppress this from outside the module (which is then sensitive to
  # the contents, detect the case in this module and suppress it explicitly.
  # Transitively, FindTclStub includes FindTCL.
  set(FPHSA_NAME_MISMATCHED 1)
endif ()
find_package_handle_standard_args(Tclsh
                                  REQUIRED_VARS TCL_TCLSH
                                  VERSION_VAR Tclsh_VERSION)
unset(FPHSA_NAME_MISMATCHED)

mark_as_advanced(TCL_TCLSH)
