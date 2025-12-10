# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindTCL
-------

Finds the Tcl (Tool Command Language), dynamic programming language:

.. code-block:: cmake

  find_package(TCL [...])

This module locates a Tcl installation, including its include directories and
libraries, and determines the appropriate Tcl library name for linking.  As part
of the Tcl ecosystem, it also finds Tk, a GUI toolkit that provides a library of
basic widgets for building graphical user interfaces.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``TCL_FOUND``
  Boolean indicating whether Tcl was found.
``TK_FOUND``
  Boolean indicating whether Tk was found.
``TCLTK_FOUND``
  Boolean indicating whether both Tcl and Tk were found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``TCL_LIBRARY``
  The path to the Tcl library (e.g., ``tcl``, etc.).
``TCL_INCLUDE_PATH``
  The directory containing ``tcl.h`` and other Tcl-related headers needed to use
  Tcl.
``TCL_TCLSH``
  The path to the ``tclsh`` command-line executable.
``TK_LIBRARY``
  The path to the Tk library (e.g., ``tk``, etc.).
``TK_INCLUDE_PATH``
  The directory containing ``tk.h`` and other Tk-related headers needed to use
  Tk.
``TK_WISH``
  The path to the ``wish`` windowing shell command-line executable.

Other Libraries
^^^^^^^^^^^^^^^

The Tcl Stub Library can be found using the separate :module:`FindTclStub`
module.

Examples
^^^^^^^^

Finding Tcl:

.. code-block:: cmake

  find_package(TCL)

See Also
^^^^^^^^

* The :module:`FindTclsh` module to find the Tcl shell command-line executable.
* The :module:`FindTclStub` module to find the Tcl Stubs Library.
* The :module:`FindWish` module to find the ``wish`` windowing shell
  command-line executable .
#]=======================================================================]

block(SCOPE_FOR POLICIES)
  cmake_policy(SET CMP0173 OLD)
  include(${CMAKE_CURRENT_LIST_DIR}/CMakeFindFrameworks.cmake)
endblock()
include(${CMAKE_CURRENT_LIST_DIR}/FindTclsh.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindWish.cmake)

if(Tclsh_VERSION)
  set(TCL_TCLSH_VERSION "${Tclsh_VERSION}")
else()
  get_filename_component(TCL_TCLSH_PATH "${TCL_TCLSH}" PATH)
  get_filename_component(TCL_TCLSH_PATH_PARENT "${TCL_TCLSH_PATH}" PATH)
  string(REGEX REPLACE
    "^.*tclsh([0-9]\\.*[0-9]).*$" "\\1" TCL_TCLSH_VERSION "${TCL_TCLSH}")
endif()

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

set(TCLTK_POSSIBLE_LIB_PATH_SUFFIXES
  lib/tcl/tcl8.7
  lib/tcl/tk8.7
  lib/tcl/tcl8.6
  lib/tcl/tk8.6
  lib/tcl/tcl8.5
  lib/tcl/tk8.5
  lib/tcl/tcl8.4
  lib/tcl/tk8.4
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

find_library(TCL_LIBRARY
  NAMES
  tcl
  tcl${TCL_LIBRARY_VERSION} tcl${TCL_TCLSH_VERSION} tcl${TK_WISH_VERSION}
  tcl87 tcl8.7 tcl87t tcl8.7t
  tcl86 tcl8.6 tcl86t tcl8.6t
  tcl85 tcl8.5
  tcl84 tcl8.4
  tcl83 tcl8.3
  tcl82 tcl8.2
  tcl80 tcl8.0
  PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
  PATH_SUFFIXES ${TCLTK_POSSIBLE_LIB_PATH_SUFFIXES}
  )

find_library(TK_LIBRARY
  NAMES
  tk
  tk${TK_LIBRARY_VERSION} tk${TCL_TCLSH_VERSION} tk${TK_WISH_VERSION}
  tk87 tk8.7 tk87t tk8.7t
  tk86 tk8.6 tk86t tk8.6t
  tk85 tk8.5
  tk84 tk8.4
  tk83 tk8.3
  tk82 tk8.2
  tk80 tk8.0
  PATHS ${TCLTK_POSSIBLE_LIB_PATHS}
  PATH_SUFFIXES ${TCLTK_POSSIBLE_LIB_PATH_SUFFIXES}
  )

CMAKE_FIND_FRAMEWORKS(Tcl)
CMAKE_FIND_FRAMEWORKS(Tk)

set(TCL_FRAMEWORK_INCLUDES)
if(Tcl_FRAMEWORKS)
  if(NOT TCL_INCLUDE_PATH)
    foreach(dir ${Tcl_FRAMEWORKS})
      set(TCL_FRAMEWORK_INCLUDES ${TCL_FRAMEWORK_INCLUDES} ${dir}/Headers)
    endforeach()
  endif()
endif()

set(TK_FRAMEWORK_INCLUDES)
if(Tk_FRAMEWORKS)
  if(NOT TK_INCLUDE_PATH)
    foreach(dir ${Tk_FRAMEWORKS})
      set(TK_FRAMEWORK_INCLUDES ${TK_FRAMEWORK_INCLUDES}
        ${dir}/Headers ${dir}/PrivateHeaders)
    endforeach()
  endif()
endif()

set(TCLTK_POSSIBLE_INCLUDE_PATHS
  "${TCL_LIBRARY_PATH_PARENT}/include"
  "${TK_LIBRARY_PATH_PARENT}/include"
  "${TCL_INCLUDE_PATH}"
  "${TK_INCLUDE_PATH}"
  ${TCL_FRAMEWORK_INCLUDES}
  ${TK_FRAMEWORK_INCLUDES}
  "${TCL_TCLSH_PATH_PARENT}/include"
  "${TK_WISH_PATH_PARENT}/include"
  )

set(TCLTK_POSSIBLE_INCLUDE_PATH_SUFFIXES
  include/tcl${TK_LIBRARY_VERSION}
  include/tcl${TCL_LIBRARY_VERSION}
  include/tcl8.7
  include/tk8.7
  include/tcl8.6
  include/tk8.6
  include/tcl8.5
  include/tk8.5
  include/tcl8.4
  include/tk8.4
  include/tcl8.3
  include/tcl8.2
  include/tcl8.0
  )

if(WIN32)
  set(TCLTK_POSSIBLE_INCLUDE_PATHS ${TCLTK_POSSIBLE_INCLUDE_PATHS}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\ActiveState\\ActiveTcl\\${ActiveTcl_CurrentVersion}]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.6;Root]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.5;Root]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.4;Root]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.3;Root]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.2;Root]/include"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Scriptics\\Tcl\\8.0;Root]/include"
    "$ENV{ProgramFiles}/Tcl/include"
    "C:/Program Files/Tcl/include"
    "C:/Tcl/include"
    )
endif()

find_path(TCL_INCLUDE_PATH
  NAMES tcl.h
  HINTS ${TCLTK_POSSIBLE_INCLUDE_PATHS}
  PATH_SUFFIXES ${TCLTK_POSSIBLE_INCLUDE_PATH_SUFFIXES}
  )

find_path(TK_INCLUDE_PATH
  NAMES tk.h
  HINTS ${TCLTK_POSSIBLE_INCLUDE_PATHS}
  PATH_SUFFIXES ${TCLTK_POSSIBLE_INCLUDE_PATH_SUFFIXES}
  )

include(FindPackageHandleStandardArgs)

if (CMAKE_FIND_PACKAGE_NAME STREQUAL "TclStub")
  # FindTclStub include()'s this module. It's an old pattern, but rather than
  # trying to suppress this from outside the module (which is then sensitive to
  # the contents, detect the case in this module and suppress it explicitly.
  set(FPHSA_NAME_MISMATCHED 1)
endif ()
find_package_handle_standard_args(TCL DEFAULT_MSG TCL_LIBRARY TCL_INCLUDE_PATH)
set(FPHSA_NAME_MISMATCHED 1)
set(TCLTK_FIND_REQUIRED ${TCL_FIND_REQUIRED})
set(TCLTK_FIND_QUIETLY  ${TCL_FIND_QUIETLY})
find_package_handle_standard_args(TCLTK DEFAULT_MSG TCL_LIBRARY TCL_INCLUDE_PATH TK_LIBRARY TK_INCLUDE_PATH)
set(TK_FIND_REQUIRED ${TCL_FIND_REQUIRED})
set(TK_FIND_QUIETLY  ${TCL_FIND_QUIETLY})
find_package_handle_standard_args(TK DEFAULT_MSG TK_LIBRARY TK_INCLUDE_PATH)
unset(FPHSA_NAME_MISMATCHED)

mark_as_advanced(
  TCL_INCLUDE_PATH
  TK_INCLUDE_PATH
  TCL_LIBRARY
  TK_LIBRARY
  )
