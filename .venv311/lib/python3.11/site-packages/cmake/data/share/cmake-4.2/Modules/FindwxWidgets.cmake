# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindwxWidgets
-------------

Finds a wxWidgets installation and provides usage requirements for usage in
projects:

.. code-block:: cmake

  find_package(wxWidgets [<version>] [COMPONENTS <components>...] [...])

wxWidgets (formerly known as wxWindows) is a widget toolkit and tools
library for creating graphical user interfaces (GUIs) for cross-platform
applications.

.. versionadded:: 3.4
  Support for :command:`find_package` version argument.

.. versionadded:: 3.14
  ``OPTIONAL_COMPONENTS`` support.

Components
^^^^^^^^^^

wxWidgets is a modular library.  This module supports components to specify
the modules to use.  Components can be specified with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(
    wxWidgets
    [COMPONENTS <components>...]
    [OPTIONAL_COMPONENTS <components>...]
  )

Supported components include:

``base``
  Finds the library that provides mandatory classes that any wxWidgets code
  depends on.  This component is always required for applications
  implementing wxWidgets.

``core``
  Finds the library that provides basic GUI classes such as GDI classes or
  controls.

``gl``
  Finds the OpenGL support.

``mono``
  Finds the wxWidgets monolithic library.

``aui``
  Finds the Advanced User Interface docking library.

``net``
  Finds the library that provides network access.

``webview``
  .. versionadded:: 3.4

  Finds the library that provides rendering of web documents
  (HTML/CSS/JavaScript).

For a full list of supported wxWidgets components, refer to the upstream
documentation.

If no components are specified, this module by default searches for ``core``
and ``base`` components.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``wxWidgets::wxWidgets``
  .. versionadded:: 3.27

  An interface imported target encapsulating the wxWidgets usage requirements
  for the found components, available if wxWidgets is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``wxWidgets_FOUND``
  Boolean indicating whether (the requested version of) wxWidgets and all
  its requested components were found.

``wxWidgets_VERSION``
  .. versionadded:: 4.2

  The version of the wxWidgets found.

``wxWidgets_INCLUDE_DIRS``
  Include directories for WIN32, i.e., where to find ``<wx/wx.h>`` and
  ``<wx/setup.h>``; possibly empty for Unix-like systems.

``wxWidgets_LIBRARIES``
  Path to the wxWidgets libraries.

``wxWidgets_LIBRARY_DIRS``
  Compile time link dirs, useful for setting ``rpath`` on Unix-like systems.
  Typically an empty string in WIN32 environment.

``wxWidgets_DEFINITIONS``
  Contains compile definitions required to compile/link against WX, e.g.
  ``WXUSINGDLL``.

``wxWidgets_DEFINITIONS_DEBUG``
  Contains compile definitions required to compile/link against WX debug builds,
  e.g. ``__WXDEBUG__``.

``wxWidgets_CXX_FLAGS``
  Include directories and compiler flags for Unix-like systems, empty on
  Windows. Essentially the output of ``wx-config --cxxflags``.

Hints
^^^^^

This module accepts the following variables before calling
``find_package(wxWidgets)``:

``WX_CONFIG``
  .. versionadded:: 3.11

  Environment variable to manually specify the name of the wxWidgets library
  configuration provider executable that will be searched besides the default
  name ``wx-config``.

``WXRC_CMD``
  .. versionadded:: 3.11

  Environment variable to manually specify the name of the wxWidgets resource
  file compiler executable that will be searched besides the default name
  ``wxrc``.

There are two search branches: a Windows style and a Unix style.  For
Windows, the following variables are searched for and set to defaults
in case of multiple choices.  Change them if the defaults are not
desired (i.e., these are the only variables that should be changed to
select a configuration):

``wxWidgets_ROOT_DIR``
  Base wxWidgets directory (e.g., ``C:/wxWidgets-3.2.0``).

``wxWidgets_LIB_DIR``
  Path to wxWidgets libraries (e.g., ``C:/wxWidgets-3.2.0/lib/vc_x64_lib``).

``wxWidgets_CONFIGURATION``
  Configuration to use (e.g., msw, mswd, mswu, mswunivud, etc.)

``wxWidgets_EXCLUDE_COMMON_LIBRARIES``
  Set to TRUE to exclude linking of commonly required libs (e.g., png, tiff,
  jpeg, zlib, webp, regex, expat, scintilla, lexilla, etc.).

For Unix style this module uses the ``wx-config`` utility.  Selecting
between debug/release, unicode/ansi, universal/non-universal, and
static/shared is possible in the QtDialog or ccmake interfaces by turning
ON/OFF the following variables:

``wxWidgets_USE_DEBUG``
  If enabled, the wxWidgets debug build will be searched.

``wxWidgets_USE_UNICODE``
  If enabled, the wxWidgets unicode build will be searched.

``wxWidgets_USE_UNIVERSAL``
  If enabled, the wxWidgets universal build will be searched.

``wxWidgets_USE_STATIC``
  If enabled, static wxWidgets libraries will be linked.

``wxWidgets_CONFIG_OPTIONS``
  This variable can be used for all other options that need to be passed to
  the wx-config utility.  For example, to use the base toolkit found on the
  system at ``/usr`` install prefix, set the variable (before calling the
  :command:`find_package` command) as such:

  .. code-block:: cmake

    set(wxWidgets_CONFIG_OPTIONS --toolkit=base --prefix=/usr)

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``wxWidgets_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``wxWidgets_VERSION``, which has the same value.

  .. versionadded:: 3.4

  The version of the wxWidgets found.

``wxWidgets_USE_FILE``
  .. deprecated:: 4.2
    Instead of using this variable, include the :module:`UsewxWidgets`
    module directly:

    .. code-block:: cmake

      include(UsewxWidgets)

  The path to the :module:`UsewxWidgets` module for using wxWidgets in the
  current directory.  For example:

  .. code-block:: cmake

    find_package(wxWidgets)
    if(wxWidgets_FOUND)
      include(${wxWidgets_USE_FILE})
    endif()

Examples
^^^^^^^^

Example: Finding wxWidgets
""""""""""""""""""""""""""

Finding wxWidgets and making it required (if wxWidgets is not found,
processing stops with an error message):

.. code-block:: cmake

   find_package(wxWidgets REQUIRED)

Example: Using Imported Target
""""""""""""""""""""""""""""""

Finding wxWidgets and using imported target in a project:

.. code-block:: cmake

  find_package(wxWidgets)
  target_link_libraries(example PRIVATE wxWidgets::wxWidgets)

Example: Using Components
"""""""""""""""""""""""""

Finding wxWidgets and specifying components:

.. code-block:: cmake

  find_package(wxWidgets COMPONENTS gl core base OPTIONAL_COMPONENTS net)
  target_link_libraries(example PRIVATE wxWidgets::wxWidgets)

Example: Monolithic wxWidgets Build
"""""""""""""""""""""""""""""""""""

Sample usage with monolithic wxWidgets build:

.. code-block:: cmake

  find_package(wxWidgets COMPONENTS mono)
  target_link_libraries(example PRIVATE wxWidgets::wxWidgets)

Example: Using Variables
""""""""""""""""""""""""

Finding and using wxWidgets in CMake versions prior to 3.27, when the
imported target wasn't yet available:

.. code-block:: cmake

  # Note that for MinGW users the order of libs is important.
  find_package(wxWidgets COMPONENTS gl core base OPTIONAL_COMPONENTS net)

  if(wxWidgets_FOUND)
    include(UsewxWidgets)
    # and for each of the project dependent executable/library targets:
    target_link_libraries(example ${wxWidgets_LIBRARIES})
  endif()
#]=======================================================================]

# NOTES
#
# This module has been tested on the WIN32 platform with wxWidgets
# 2.6.2, 2.6.3, and 2.5.3. However, it has been designed to
# easily extend support to all possible builds, e.g., static/shared,
# debug/release, unicode, universal, multilib/monolithic, etc..
#
# If you want to use the module and your build type is not supported
# out-of-the-box, please contact me to exchange information on how
# your system is setup and I'll try to add support for it.
#
# AUTHOR
#
# Miguel A. Figueroa-Villanueva (miguelf at ieee dot org).
# Jan Woetzel (jw at mip.informatik.uni-kiel.de).
#
# Based on previous works of:
# Jan Woetzel (FindwxWindows.cmake),
# Jorgen Bodde and Jerry Fath (FindwxWin.cmake).

# TODO/ideas
#
# (1) Option/Setting to use all available wx libs
# In contrast to expert developer who lists the
# minimal set of required libs in wxWidgets_USE_LIBS
# there is the newbie user:
#   - who just wants to link against WX with more 'magic'
#   - doesn't know the internal structure of WX or how it was built,
#     in particular if it is monolithic or not
#   - want to link against all available WX libs
# Basically, the intent here is to mimic what wx-config would do by
# default (i.e., `wx-config --libs`).
#
# Possible solution:
#   Add a reserved keyword "std" that initializes to what wx-config
# would default to. If the user has not set the wxWidgets_USE_LIBS,
# default to "std" instead of "base core" as it is now. To implement
# "std" will basically boil down to a FOR_EACH lib-FOUND, but maybe
# checking whether a minimal set was found.


# FIXME: This and all the DBG_MSG calls should be removed after the
# module stabilizes.
#
# Helper macro to control the debugging output globally. There are
# two versions for controlling how verbose your output should be.
macro(DBG_MSG _MSG)
#  message(STATUS
#    "${CMAKE_CURRENT_LIST_FILE}(${CMAKE_CURRENT_LIST_LINE}): ${_MSG}")
endmacro()
macro(DBG_MSG_V _MSG)
#  message(STATUS
#    "${CMAKE_CURRENT_LIST_FILE}(${CMAKE_CURRENT_LIST_LINE}): ${_MSG}")
endmacro()

# Clear return values in case the module is loaded more than once.
set(wxWidgets_FOUND FALSE)
set(wxWidgets_INCLUDE_DIRS "")
set(wxWidgets_LIBRARIES    "")
set(wxWidgets_LIBRARY_DIRS "")
set(wxWidgets_CXX_FLAGS    "")

# DEPRECATED: This is a patch to support the DEPRECATED use of
# wxWidgets_USE_LIBS.
#
# If wxWidgets_USE_LIBS is set:
# - if using <components>, then override wxWidgets_USE_LIBS
# - else set wxWidgets_FIND_COMPONENTS to wxWidgets_USE_LIBS
if(wxWidgets_USE_LIBS AND NOT wxWidgets_FIND_COMPONENTS)
  set(wxWidgets_FIND_COMPONENTS ${wxWidgets_USE_LIBS})
endif()
DBG_MSG("wxWidgets_FIND_COMPONENTS : ${wxWidgets_FIND_COMPONENTS}")

# Add the convenience use file if available.
#
# Get dir of this file which may reside in:
# - CMAKE_ROOT/Modules on CMake installation
# - CMAKE_MODULE_PATH if the user prefers their own specialized version
set(wxWidgets_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_DIR}")
# Prefer an existing customized version, but the user might override
# the FindwxWidgets module and not the UsewxWidgets one.
if(EXISTS "${wxWidgets_CURRENT_LIST_DIR}/UsewxWidgets.cmake")
  set(wxWidgets_USE_FILE "${wxWidgets_CURRENT_LIST_DIR}/UsewxWidgets.cmake")
else()
  set(wxWidgets_USE_FILE UsewxWidgets)
endif()

# Known wxWidgets versions.
set(wx_versions 3.3 3.2 3.1 3.0 2.9 2.8 2.7 2.6 2.5)

macro(wx_extract_version)
  unset(_wx_filename)
  find_file(_wx_filename wx/version.h PATHS ${wxWidgets_INCLUDE_DIRS} NO_DEFAULT_PATH)
  dbg_msg("_wx_filename:  ${_wx_filename}")

  if(NOT _wx_filename)
    message(FATAL_ERROR "wxWidgets wx/version.h file not found in ${wxWidgets_INCLUDE_DIRS}.")
  endif()

  file(READ "${_wx_filename}" _wx_version_h)
  unset(_wx_filename CACHE)

  string(REGEX REPLACE "^(.*\n)?#define +wxMAJOR_VERSION +([0-9]+).*"
    "\\2" wxWidgets_VERSION_MAJOR "${_wx_version_h}" )
  string(REGEX REPLACE "^(.*\n)?#define +wxMINOR_VERSION +([0-9]+).*"
    "\\2" wxWidgets_VERSION_MINOR "${_wx_version_h}" )
  string(REGEX REPLACE "^(.*\n)?#define +wxRELEASE_NUMBER +([0-9]+).*"
    "\\2" wxWidgets_VERSION_PATCH "${_wx_version_h}" )
  string(REGEX REPLACE "^(.*\n)?#define +wxSUBRELEASE_NUMBER +([0-9]+).*"
    "\\2" wxWidgets_VERSION_TWEAK "${_wx_version_h}" )

  set(wxWidgets_VERSION
    "${wxWidgets_VERSION_MAJOR}.${wxWidgets_VERSION_MINOR}.${wxWidgets_VERSION_PATCH}")
  if(${wxWidgets_VERSION_TWEAK} GREATER 0)
    string(APPEND wxWidgets_VERSION ".${wxWidgets_VERSION_TWEAK}")
  endif()
  set(wxWidgets_VERSION_STRING "${wxWidgets_VERSION}")
endmacro()

#=====================================================================
# Determine whether unix or win32 paths should be used
#=====================================================================
if(WIN32 AND NOT CYGWIN AND NOT MSYS AND NOT CMAKE_CROSSCOMPILING)
  set(wxWidgets_FIND_STYLE "win32")
else()
  set(wxWidgets_FIND_STYLE "unix")
endif()

#=====================================================================
# WIN32_FIND_STYLE
#=====================================================================
if(wxWidgets_FIND_STYLE STREQUAL "win32")
  # Useful common wx libs needed by almost all components.
  set(wxWidgets_WEBP_LIBRARIES webp webpdemux sharpyuv)
  set(wxWidgets_COMMON_LIBRARIES png tiff jpeg zlib ${wxWidgets_WEBP_LIBRARIES} regex expat)

  # Libraries needed by stc component
  set(wxWidgets_STC_LIBRARIES scintilla lexilla)

  # DEPRECATED: Use find_package(wxWidgets COMPONENTS mono) instead.
  if(NOT wxWidgets_FIND_COMPONENTS)
    if(wxWidgets_USE_MONOLITHIC)
      set(wxWidgets_FIND_COMPONENTS mono)
    else()
      set(wxWidgets_FIND_COMPONENTS core base) # this is default
    endif()
  endif()

  # Add the common (usually required libs) unless
  # wxWidgets_EXCLUDE_COMMON_LIBRARIES has been set.
  if(NOT wxWidgets_EXCLUDE_COMMON_LIBRARIES)
    if(stc IN_LIST wxWidgets_FIND_COMPONENTS)
      list(APPEND wxWidgets_FIND_COMPONENTS ${wxWidgets_STC_LIBRARIES})
    endif()
    list(APPEND wxWidgets_FIND_COMPONENTS ${wxWidgets_COMMON_LIBRARIES})
  endif()

  # Remove duplicates, for example when user has specified common libraries.
  list(REMOVE_DUPLICATES wxWidgets_FIND_COMPONENTS)

  #-------------------------------------------------------------------
  # WIN32: Helper MACROS
  #-------------------------------------------------------------------
  #
  # Get filename components for a configuration. For example,
  #   if _CONFIGURATION = mswunivud, then _PF="msw", _UNV=univ, _UCD=u _DBG=d
  #   if _CONFIGURATION = mswu,      then _PF="msw", _UNV="",   _UCD=u _DBG=""
  #
  macro(WX_GET_NAME_COMPONENTS _CONFIGURATION _PF _UNV _UCD _DBG)
    DBG_MSG_V(${_CONFIGURATION})
    string(REGEX MATCH "univ" ${_UNV} "${_CONFIGURATION}")
    string(REGEX REPLACE "[msw|qt].*(u)[d]*$" "u" ${_UCD} "${_CONFIGURATION}")
    if(${_UCD} STREQUAL ${_CONFIGURATION})
      set(${_UCD} "")
    endif()
    string(REGEX MATCH "d$" ${_DBG} "${_CONFIGURATION}")
    string(REGEX MATCH "^[msw|qt]*" ${_PF} "${_CONFIGURATION}")
  endmacro()

  #
  # Find libraries associated to a configuration.
  #
  macro(WX_FIND_LIBS _PF _UNV _UCD _DBG _VER)
    DBG_MSG_V("m_unv = ${_UNV}")
    DBG_MSG_V("m_ucd = ${_UCD}")
    DBG_MSG_V("m_dbg = ${_DBG}")
    DBG_MSG_V("m_ver = ${_VER}")

    # FIXME: What if both regex libs are available. regex should be
    # found outside the loop and only wx${LIB}${_UCD}${_DBG}.
    # Find wxWidgets common libraries.
    foreach(LIB ${wxWidgets_COMMON_LIBRARIES} ${wxWidgets_STC_LIBRARIES})
      find_library(WX_${LIB}${_DBG}
        NAMES
        wx${LIB}${_UCD}${_DBG} # for regex
        wx${LIB}${_DBG}
        PATHS ${WX_LIB_DIR}
        NO_DEFAULT_PATH
        )
      mark_as_advanced(WX_${LIB}${_DBG})
    endforeach()

    # Find wxWidgets multilib base libraries.
    find_library(WX_base${_DBG}
      NAMES wxbase${_VER}${_UCD}${_DBG}
      PATHS ${WX_LIB_DIR}
      NO_DEFAULT_PATH
      )
    mark_as_advanced(WX_base${_DBG})
    foreach(LIB net odbc xml)
      find_library(WX_${LIB}${_DBG}
        NAMES wxbase${_VER}${_UCD}${_DBG}_${LIB}
        PATHS ${WX_LIB_DIR}
        NO_DEFAULT_PATH
        )
      mark_as_advanced(WX_${LIB}${_DBG})
    endforeach()

    # Find wxWidgets monolithic library.
    find_library(WX_mono${_DBG}
      NAMES wx${_PF}${_UNV}${_VER}${_UCD}${_DBG}
      PATHS ${WX_LIB_DIR}
      NO_DEFAULT_PATH
      )
    mark_as_advanced(WX_mono${_DBG})

    # Find wxWidgets multilib libraries.
    foreach(LIB core adv aui html media xrc dbgrid gl qa richtext
                stc ribbon propgrid webview)
      find_library(WX_${LIB}${_DBG}
        NAMES wx${_PF}${_UNV}${_VER}${_UCD}${_DBG}_${LIB}
        PATHS ${WX_LIB_DIR}
        NO_DEFAULT_PATH
        )
      mark_as_advanced(WX_${LIB}${_DBG})
    endforeach()
  endmacro()

  #
  # Clear all library paths, so that FIND_LIBRARY refinds them.
  #
  # Clear a lib, reset its found flag, and mark as advanced.
  macro(WX_CLEAR_LIB _LIB)
    set(${_LIB} "${_LIB}-NOTFOUND" CACHE FILEPATH "Cleared." FORCE)
    set(${_LIB}_FOUND FALSE)
    mark_as_advanced(${_LIB})
  endmacro()
  # Clear all debug or release library paths (arguments are "d" or "").
  macro(WX_CLEAR_ALL_LIBS _DBG)
    # Clear wxWidgets common libraries.
    foreach(LIB ${wxWidgets_COMMON_LIBRARIES} ${wxWidgets_STC_LIBRARIES})
      WX_CLEAR_LIB(WX_${LIB}${_DBG})
    endforeach()

    # Clear wxWidgets multilib base libraries.
    WX_CLEAR_LIB(WX_base${_DBG})
    foreach(LIB net odbc xml)
      WX_CLEAR_LIB(WX_${LIB}${_DBG})
    endforeach()

    # Clear wxWidgets monolithic library.
    WX_CLEAR_LIB(WX_mono${_DBG})

    # Clear wxWidgets multilib libraries.
    foreach(LIB core adv aui html media xrc dbgrid gl qa richtext
                webview stc ribbon propgrid)
      WX_CLEAR_LIB(WX_${LIB}${_DBG})
    endforeach()
  endmacro()
  # Clear all wxWidgets debug libraries.
  macro(WX_CLEAR_ALL_DBG_LIBS)
    WX_CLEAR_ALL_LIBS("d")
  endmacro()
  # Clear all wxWidgets release libraries.
  macro(WX_CLEAR_ALL_REL_LIBS)
    WX_CLEAR_ALL_LIBS("")
  endmacro()

  #
  # Set the wxWidgets_LIBRARIES variable.
  # Also, Sets output variable wxWidgets_FOUND to FALSE if it fails.
  #
  macro(WX_SET_LIBRARIES _LIBS _DBG)
    DBG_MSG_V("Looking for ${${_LIBS}}")
    if(WX_USE_REL_AND_DBG)
      foreach(LIB ${${_LIBS}})
        DBG_MSG_V("Searching for ${LIB} and ${LIB}d")
        DBG_MSG_V("WX_${LIB}  : ${WX_${LIB}}")
        DBG_MSG_V("WX_${LIB}d : ${WX_${LIB}d}")
        if(WX_${LIB} AND WX_${LIB}d)
          DBG_MSG_V("Found ${LIB} and ${LIB}d")
          list(APPEND wxWidgets_LIBRARIES
            debug ${WX_${LIB}d} optimized ${WX_${LIB}}
            )
          set(wxWidgets_${LIB}_FOUND TRUE)
        elseif(NOT wxWidgets_FIND_REQUIRED_${LIB})
          DBG_MSG_V("- ignored optional missing WX_${LIB}=${WX_${LIB}} or WX_${LIB}d=${WX_${LIB}d}")
        else()
          DBG_MSG_V("- not found due to missing WX_${LIB}=${WX_${LIB}} or WX_${LIB}d=${WX_${LIB}d}")
          set(wxWidgets_FOUND FALSE)
        endif()
      endforeach()
    else()
      foreach(LIB ${${_LIBS}})
        DBG_MSG_V("Searching for ${LIB}${_DBG}")
        DBG_MSG_V("WX_${LIB}${_DBG} : ${WX_${LIB}${_DBG}}")
        if(WX_${LIB}${_DBG})
          DBG_MSG_V("Found ${LIB}${_DBG}")
          list(APPEND wxWidgets_LIBRARIES ${WX_${LIB}${_DBG}})
          set(wxWidgets_${LIB}_FOUND TRUE)
        elseif(NOT wxWidgets_FIND_REQUIRED_${LIB})
          DBG_MSG_V("- ignored optional missing WX_${LIB}${_DBG}=${WX_${LIB}${_DBG}}")
        else()
          DBG_MSG_V("- not found due to missing WX_${LIB}${_DBG}=${WX_${LIB}${_DBG}}")
          set(wxWidgets_FOUND FALSE)
        endif()
      endforeach()
    endif()

    DBG_MSG_V("OpenGL")
    if(gl IN_LIST ${_LIBS})
      DBG_MSG_V("- is required.")
      list(APPEND wxWidgets_LIBRARIES opengl32 glu32)
    endif()

    if(stc IN_LIST ${_LIBS})
      list(APPEND wxWidgets_LIBRARIES imm32)
    endif()

    list(APPEND wxWidgets_LIBRARIES
      kernel32
      user32
      gdi32
      gdiplus
      msimg32
      comdlg32
      winspool
      winmm
      shell32
      shlwapi
      comctl32
      ole32
      oleaut32
      uuid
      rpcrt4
      advapi32
      version
      ws2_32
      wininet
      oleacc
      uxtheme
      wsock32
    )
  endmacro()

  #-------------------------------------------------------------------
  # WIN32: Start actual work.
  #-------------------------------------------------------------------

  set(wx_paths "wxWidgets")
  foreach(version ${wx_versions})
    foreach(patch RANGE 15 0 -1)
      list(APPEND wx_paths "wxWidgets-${version}.${patch}")
      foreach(tweak RANGE 3 1 -1)
        list(APPEND wx_paths "wxWidgets-${version}.${patch}.${tweak}")
      endforeach()
    endforeach()
  endforeach()

  # Look for an installation tree.
  find_path(wxWidgets_ROOT_DIR
    NAMES include/wx/wx.h
    PATHS
      ENV wxWidgets_ROOT_DIR
      ENV WXWIN
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\wxWidgets_is1;Inno Setup: App Path]"  # WX 2.6.x
      C:/
      D:/
      ENV ProgramFiles
    PATH_SUFFIXES
      ${wx_paths}
    DOC "wxWidgets base/installation directory"
    )

  # If wxWidgets_ROOT_DIR changed, clear lib dir.
  if(NOT WX_ROOT_DIR STREQUAL wxWidgets_ROOT_DIR)
    if(NOT wxWidgets_LIB_DIR OR WX_ROOT_DIR)
      set(wxWidgets_LIB_DIR "wxWidgets_LIB_DIR-NOTFOUND"
          CACHE PATH "Cleared." FORCE)
    endif()
    set(WX_ROOT_DIR ${wxWidgets_ROOT_DIR}
        CACHE INTERNAL "wxWidgets_ROOT_DIR")
  endif()

  if(WX_ROOT_DIR)
    # Select one default tree inside the already determined wx tree.
    # Prefer static/shared order usually consistent with build
    # settings.
    set(_WX_TOOL "")
    set(_WX_TOOLVER "")
    set(_WX_ARCH "")
    if(MINGW)
      set(_WX_TOOL gcc)
    elseif(MSVC)
      set(_WX_TOOL vc)
      set(_WX_TOOLVER ${MSVC_TOOLSET_VERSION})
      # support for a lib/vc14x_x64_dll/ path from wxW 3.1.3 distribution
      string(REGEX REPLACE ".$" "x" _WX_TOOLVERx ${_WX_TOOLVER})
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_WX_ARCH _x64)
      endif()
    endif()
    if(BUILD_SHARED_LIBS)
      find_path(wxWidgets_LIB_DIR
        NAMES
          qtu/wx/setup.h
          qtud/wx/setup.h
          msw/wx/setup.h
          mswd/wx/setup.h
          mswu/wx/setup.h
          mswud/wx/setup.h
          mswuniv/wx/setup.h
          mswunivd/wx/setup.h
          mswunivu/wx/setup.h
          mswunivud/wx/setup.h
        PATHS
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}_xp${_WX_ARCH}_dll   # prefer shared
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}${_WX_ARCH}_dll   # prefer shared
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}_xp${_WX_ARCH}_dll   # prefer shared
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}${_WX_ARCH}_dll   # prefer shared
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_ARCH}_dll                 # prefer shared
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}_xp${_WX_ARCH}_lib
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}${_WX_ARCH}_lib
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}_xp${_WX_ARCH}_lib
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}${_WX_ARCH}_lib
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_ARCH}_lib
        DOC "Path to wxWidgets libraries"
        NO_DEFAULT_PATH
        )
    else()
      find_path(wxWidgets_LIB_DIR
        NAMES
          qtu/wx/setup.h
          qtud/wx/setup.h
          msw/wx/setup.h
          mswd/wx/setup.h
          mswu/wx/setup.h
          mswud/wx/setup.h
          mswuniv/wx/setup.h
          mswunivd/wx/setup.h
          mswunivu/wx/setup.h
          mswunivud/wx/setup.h
        PATHS
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}_xp${_WX_ARCH}_lib   # prefer static
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}${_WX_ARCH}_lib   # prefer static
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}_xp${_WX_ARCH}_lib   # prefer static
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}${_WX_ARCH}_lib   # prefer static
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_ARCH}_lib                 # prefer static
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}_xp${_WX_ARCH}_dll
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVER}${_WX_ARCH}_dll
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}_xp${_WX_ARCH}_dll
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_TOOLVERx}${_WX_ARCH}_dll
        ${WX_ROOT_DIR}/lib/${_WX_TOOL}${_WX_ARCH}_dll
        DOC "Path to wxWidgets libraries"
        NO_DEFAULT_PATH
        )
    endif()
    unset(_WX_TOOL)
    unset(_WX_TOOLVER)
    unset(_WX_ARCH)

    # If wxWidgets_LIB_DIR changed, clear all libraries.
    if(NOT WX_LIB_DIR STREQUAL wxWidgets_LIB_DIR)
      set(WX_LIB_DIR ${wxWidgets_LIB_DIR} CACHE INTERNAL "wxWidgets_LIB_DIR")
      WX_CLEAR_ALL_DBG_LIBS()
      WX_CLEAR_ALL_REL_LIBS()
    endif()

    if(WX_LIB_DIR)
      # If building shared libs, define WXUSINGDLL to use dllimport.
      if(WX_LIB_DIR MATCHES "[dD][lL][lL]")
        set(wxWidgets_DEFINITIONS WXUSINGDLL)
        DBG_MSG_V("detected SHARED/DLL tree WX_LIB_DIR=${WX_LIB_DIR}")
      endif()

      # Search for available configuration types.
      foreach(CFG mswunivud mswunivd mswud mswd mswunivu mswuniv mswu msw qt qtd qtu qtud)
        set(WX_${CFG}_FOUND FALSE)
        if(EXISTS ${WX_LIB_DIR}/${CFG})
          list(APPEND WX_CONFIGURATION_LIST ${CFG})
          set(WX_${CFG}_FOUND TRUE)
          set(WX_CONFIGURATION ${CFG})
        endif()
      endforeach()
      DBG_MSG_V("WX_CONFIGURATION_LIST=${WX_CONFIGURATION_LIST}")

      if(WX_CONFIGURATION)
        set(wxWidgets_FOUND TRUE)

        # If the selected configuration wasn't found force the default
        # one. Otherwise, use it but still force a refresh for
        # updating the doc string with the current list of available
        # configurations.
        if(NOT WX_${wxWidgets_CONFIGURATION}_FOUND)
          set(wxWidgets_CONFIGURATION ${WX_CONFIGURATION} CACHE STRING
            "Set wxWidgets configuration (${WX_CONFIGURATION_LIST})" FORCE)
        else()
          set(wxWidgets_CONFIGURATION ${wxWidgets_CONFIGURATION} CACHE STRING
            "Set wxWidgets configuration (${WX_CONFIGURATION_LIST})" FORCE)
        endif()

        # If release config selected, and both release/debug exist.
        if(WX_${wxWidgets_CONFIGURATION}d_FOUND)
          option(wxWidgets_USE_REL_AND_DBG
            "Use release and debug configurations?" TRUE)
          set(WX_USE_REL_AND_DBG ${wxWidgets_USE_REL_AND_DBG})
        else()
          # If the option exists (already in cache), force it false.
          if(wxWidgets_USE_REL_AND_DBG)
            set(wxWidgets_USE_REL_AND_DBG FALSE CACHE BOOL
              "No ${wxWidgets_CONFIGURATION}d found." FORCE)
          endif()
          set(WX_USE_REL_AND_DBG FALSE)
        endif()

        # Get configuration parameters from the name.
        WX_GET_NAME_COMPONENTS(${wxWidgets_CONFIGURATION} PF UNV UCD DBG)

        # Set wxWidgets lib setup include directory.
        if(EXISTS ${WX_LIB_DIR}/${wxWidgets_CONFIGURATION}/wx/setup.h)
          set(wxWidgets_INCLUDE_DIRS
            ${WX_LIB_DIR}/${wxWidgets_CONFIGURATION})
        else()
          DBG_MSG("wxWidgets_FOUND FALSE because ${WX_LIB_DIR}/${wxWidgets_CONFIGURATION}/wx/setup.h does not exist.")
          set(wxWidgets_FOUND FALSE)
        endif()

        # Set wxWidgets main include directory.
        if(EXISTS ${WX_ROOT_DIR}/include/wx/wx.h)
          list(APPEND wxWidgets_INCLUDE_DIRS ${WX_ROOT_DIR}/include)
        else()
          DBG_MSG("wxWidgets_FOUND FALSE because WX_ROOT_DIR=${WX_ROOT_DIR} has no ${WX_ROOT_DIR}/include/wx/wx.h")
          set(wxWidgets_FOUND FALSE)
        endif()

        # Get version number.
        wx_extract_version()
        set(VER "${wxWidgets_VERSION_MAJOR}${wxWidgets_VERSION_MINOR}")

        # Find wxWidgets libraries.
        WX_FIND_LIBS("${PF}" "${UNV}" "${UCD}" "${DBG}" "${VER}")
        if(WX_USE_REL_AND_DBG)
          WX_FIND_LIBS("${PF}" "${UNV}" "${UCD}" "d" "${VER}")
        endif()

        # Settings for requested libs (i.e., include dir, libraries, etc.).
        WX_SET_LIBRARIES(wxWidgets_FIND_COMPONENTS "${DBG}")

        # Add necessary definitions for unicode builds
        if("${UCD}" STREQUAL "u")
          list(APPEND wxWidgets_DEFINITIONS UNICODE _UNICODE)
        endif()

        # Add necessary definitions for debug builds
        set(wxWidgets_DEFINITIONS_DEBUG _DEBUG __WXDEBUG__)

      endif()
    endif()
  endif()

  if(MINGW AND NOT wxWidgets_FOUND)
    # Try unix search mode as well.
    set(wxWidgets_FIND_STYLE "unix")
    dbg_msg_v("wxWidgets_FIND_STYLE changed to unix")
  endif()
endif()

#=====================================================================
# UNIX_FIND_STYLE
#=====================================================================
if(wxWidgets_FIND_STYLE STREQUAL "unix")
    #-----------------------------------------------------------------
    # UNIX: Helper MACROS
    #-----------------------------------------------------------------
    #
    # Set the default values based on "wx-config --selected-config".
    #
    macro(WX_CONFIG_SELECT_GET_DEFAULT)
      execute_process(
        COMMAND sh "${wxWidgets_CONFIG_EXECUTABLE}"
          ${wxWidgets_CONFIG_OPTIONS} --selected-config
        OUTPUT_VARIABLE _wx_selected_config
        RESULT_VARIABLE _wx_result
        ERROR_QUIET
        )
      if(_wx_result EQUAL 0)
        foreach(_opt_name debug static unicode universal)
          string(TOUPPER ${_opt_name} _upper_opt_name)
          if(_wx_selected_config MATCHES "${_opt_name}")
            set(wxWidgets_DEFAULT_${_upper_opt_name} ON)
          else()
            set(wxWidgets_DEFAULT_${_upper_opt_name} OFF)
          endif()
        endforeach()
      else()
        foreach(_upper_opt_name DEBUG STATIC UNICODE UNIVERSAL)
          set(wxWidgets_DEFAULT_${_upper_opt_name} OFF)
        endforeach()
      endif()
    endmacro()

    #
    # Query a boolean configuration option to determine if the system
    # has both builds available. If so, provide the selection option
    # to the user.
    #
    macro(WX_CONFIG_SELECT_QUERY_BOOL _OPT_NAME _OPT_HELP)
      execute_process(
        COMMAND sh "${wxWidgets_CONFIG_EXECUTABLE}"
          ${wxWidgets_CONFIG_OPTIONS} --${_OPT_NAME}=yes
        RESULT_VARIABLE _wx_result_yes
        OUTPUT_QUIET
        ERROR_QUIET
        )
      execute_process(
        COMMAND sh "${wxWidgets_CONFIG_EXECUTABLE}"
          ${wxWidgets_CONFIG_OPTIONS} --${_OPT_NAME}=no
        RESULT_VARIABLE _wx_result_no
        OUTPUT_QUIET
        ERROR_QUIET
        )
      string(TOUPPER ${_OPT_NAME} _UPPER_OPT_NAME)
      if(_wx_result_yes EQUAL 0 AND _wx_result_no EQUAL 0)
        option(wxWidgets_USE_${_UPPER_OPT_NAME}
          ${_OPT_HELP} ${wxWidgets_DEFAULT_${_UPPER_OPT_NAME}})
      else()
        # If option exists (already in cache), force to available one.
        if(DEFINED wxWidgets_USE_${_UPPER_OPT_NAME})
          if(_wx_result_yes EQUAL 0)
            set(wxWidgets_USE_${_UPPER_OPT_NAME} ON  CACHE BOOL ${_OPT_HELP} FORCE)
          else()
            set(wxWidgets_USE_${_UPPER_OPT_NAME} OFF CACHE BOOL ${_OPT_HELP} FORCE)
          endif()
        endif()
      endif()
    endmacro()

    #
    # Set wxWidgets_SELECT_OPTIONS to wx-config options for selecting
    # among multiple builds.
    #
    macro(WX_CONFIG_SELECT_SET_OPTIONS)
      set(wxWidgets_SELECT_OPTIONS ${wxWidgets_CONFIG_OPTIONS})
      foreach(_opt_name debug static unicode universal)
        string(TOUPPER ${_opt_name} _upper_opt_name)
        if(DEFINED wxWidgets_USE_${_upper_opt_name})
          if(wxWidgets_USE_${_upper_opt_name})
            list(APPEND wxWidgets_SELECT_OPTIONS --${_opt_name}=yes)
          else()
            list(APPEND wxWidgets_SELECT_OPTIONS --${_opt_name}=no)
          endif()
        endif()
      endforeach()
    endmacro()

    #-----------------------------------------------------------------
    # UNIX: Start actual work.
    #-----------------------------------------------------------------
    # Support cross-compiling, only search in the target platform.
    #
    # Look for wx-config -- this can be set in the environment,
    # or try versioned and toolchain-versioned variants of the -config
    # executable as well.
    set(wx_config_names "wx-config")
    foreach(version ${wx_versions})
      list(APPEND wx_config_names "wx-config-${version}" "wxgtk3u-${version}-config" "wxgtk2u-${version}-config")
    endforeach()
    find_program(wxWidgets_CONFIG_EXECUTABLE
      NAMES
        $ENV{WX_CONFIG}
        ${wx_config_names}
      DOC "Location of wxWidgets library configuration provider binary (wx-config)."
      ONLY_CMAKE_FIND_ROOT_PATH
      )

    if(wxWidgets_CONFIG_EXECUTABLE)
      set(wxWidgets_FOUND TRUE)

      # get defaults based on "wx-config --selected-config"
      WX_CONFIG_SELECT_GET_DEFAULT()

      # for each option: if both builds are available, provide option
      WX_CONFIG_SELECT_QUERY_BOOL(debug "Use debug build?")
      WX_CONFIG_SELECT_QUERY_BOOL(unicode "Use unicode build?")
      WX_CONFIG_SELECT_QUERY_BOOL(universal "Use universal build?")
      WX_CONFIG_SELECT_QUERY_BOOL(static "Link libraries statically?")

      # process selection to set wxWidgets_SELECT_OPTIONS
      WX_CONFIG_SELECT_SET_OPTIONS()
      DBG_MSG("wxWidgets_SELECT_OPTIONS=${wxWidgets_SELECT_OPTIONS}")

      # run the wx-config program to get cxxflags
      execute_process(
        COMMAND sh "${wxWidgets_CONFIG_EXECUTABLE}"
          ${wxWidgets_SELECT_OPTIONS} --cxxflags
        OUTPUT_VARIABLE wxWidgets_CXX_FLAGS
        RESULT_VARIABLE RET
        ERROR_QUIET
        )
      if(RET EQUAL 0)
        string(STRIP "${wxWidgets_CXX_FLAGS}" wxWidgets_CXX_FLAGS)
        separate_arguments(wxWidgets_CXX_FLAGS_LIST NATIVE_COMMAND "${wxWidgets_CXX_FLAGS}")

        DBG_MSG_V("wxWidgets_CXX_FLAGS=${wxWidgets_CXX_FLAGS}")

        # parse definitions and include dirs from cxxflags
        #   drop the -D and -I prefixes
        set(wxWidgets_CXX_FLAGS)
        foreach(arg IN LISTS wxWidgets_CXX_FLAGS_LIST)
          if("${arg}" MATCHES "^-I(.*)$")
            # include directory
            list(APPEND wxWidgets_INCLUDE_DIRS "${CMAKE_MATCH_1}")
          elseif("${arg}" MATCHES "^-D(.*)$")
            # compile definition
            list(APPEND wxWidgets_DEFINITIONS "${CMAKE_MATCH_1}")
          else()
            list(APPEND wxWidgets_CXX_FLAGS "${arg}")
          endif()
        endforeach()

        DBG_MSG_V("wxWidgets_DEFINITIONS=${wxWidgets_DEFINITIONS}")
        DBG_MSG_V("wxWidgets_INCLUDE_DIRS=${wxWidgets_INCLUDE_DIRS}")
        DBG_MSG_V("wxWidgets_CXX_FLAGS=${wxWidgets_CXX_FLAGS}")

      else()
        set(wxWidgets_FOUND FALSE)
        DBG_MSG_V(
          "${wxWidgets_CONFIG_EXECUTABLE} --cxxflags FAILED with RET=${RET}")
      endif()

      # run the wx-config program to get the libs
      # - NOTE: wx-config doesn't verify that the libs requested exist
      #         it just produces the names. Maybe a TRY_COMPILE would
      #         be useful here...
      unset(_cmp_req)
      unset(_cmp_opt)
      foreach(_cmp IN LISTS wxWidgets_FIND_COMPONENTS)
        if(wxWidgets_FIND_REQUIRED_${_cmp})
          list(APPEND _cmp_req "${_cmp}")
        else()
          list(APPEND _cmp_opt "${_cmp}")
        endif()
      endforeach()
      DBG_MSG_V("wxWidgets required components : ${_cmp_req}")
      DBG_MSG_V("wxWidgets optional components : ${_cmp_opt}")
      if(DEFINED _cmp_opt)
        string(REPLACE ";" "," _cmp_opt "${_cmp_opt}")
        set(_cmp_opt "--optional-libs" ${_cmp_opt})
      endif()
      string(REPLACE ";" "," _cmp_req "${_cmp_req}")
      execute_process(
        COMMAND sh "${wxWidgets_CONFIG_EXECUTABLE}"
          ${wxWidgets_SELECT_OPTIONS} --libs ${_cmp_req} ${_cmp_opt}
        OUTPUT_VARIABLE wxWidgets_LIBRARIES
        RESULT_VARIABLE RET
        ERROR_QUIET
        )
      if(RET EQUAL 0)
        string(STRIP "${wxWidgets_LIBRARIES}" wxWidgets_LIBRARIES)
        separate_arguments(wxWidgets_LIBRARIES)
        string(REPLACE "-framework;" "-framework "
          wxWidgets_LIBRARIES "${wxWidgets_LIBRARIES}")
        string(REPLACE "-weak_framework;" "-weak_framework "
          wxWidgets_LIBRARIES "${wxWidgets_LIBRARIES}")
        string(REPLACE "-arch;" "-arch "
          wxWidgets_LIBRARIES "${wxWidgets_LIBRARIES}")
        string(REPLACE "-isysroot;" "-isysroot "
          wxWidgets_LIBRARIES "${wxWidgets_LIBRARIES}")

        # extract linkdirs (-L) for rpath (i.e., LINK_DIRECTORIES)
        string(REGEX MATCHALL "-L[^;]+"
          wxWidgets_LIBRARY_DIRS "${wxWidgets_LIBRARIES}")
        string(REGEX REPLACE "-L([^;]+)" "\\1"
          wxWidgets_LIBRARY_DIRS "${wxWidgets_LIBRARY_DIRS}")

        DBG_MSG_V("wxWidgets_LIBRARIES=${wxWidgets_LIBRARIES}")
        DBG_MSG_V("wxWidgets_LIBRARY_DIRS=${wxWidgets_LIBRARY_DIRS}")

      else()
        set(wxWidgets_FOUND FALSE)
        DBG_MSG("${wxWidgets_CONFIG_EXECUTABLE} --libs ${_cmp_req} ${_cmp_opt} FAILED with RET=${RET}")
      endif()
      unset(_cmp_req)
      unset(_cmp_opt)
    endif()

    # When using wx-config in MSYS, the include paths are UNIX style paths which may or may
    # not work correctly depending on you MSYS/MinGW configuration.  CMake expects native
    # paths internally.
    if(wxWidgets_FOUND AND MSYS)
      find_program(_cygpath_exe cygpath ONLY_CMAKE_FIND_ROOT_PATH)
      DBG_MSG_V("_cygpath_exe:  ${_cygpath_exe}")
      if(_cygpath_exe)
        set(_tmp_path "")
        foreach(_path ${wxWidgets_INCLUDE_DIRS})
          execute_process(
            COMMAND cygpath -w ${_path}
            OUTPUT_VARIABLE _native_path
            RESULT_VARIABLE _retv
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            )
          if(_retv EQUAL 0)
            file(TO_CMAKE_PATH ${_native_path} _native_path)
            DBG_MSG_V("Path ${_path} converted to ${_native_path}")
            string(APPEND _tmp_path " ${_native_path}")
          endif()
        endforeach()
        DBG_MSG("Setting wxWidgets_INCLUDE_DIRS = ${_tmp_path}")
        set(wxWidgets_INCLUDE_DIRS ${_tmp_path})
        separate_arguments(wxWidgets_INCLUDE_DIRS)
        list(REMOVE_ITEM wxWidgets_INCLUDE_DIRS "")

        set(_tmp_path "")
        foreach(_path ${wxWidgets_LIBRARY_DIRS})
          execute_process(
            COMMAND cygpath -w ${_path}
            OUTPUT_VARIABLE _native_path
            RESULT_VARIABLE _retv
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            )
          if(_retv EQUAL 0)
            file(TO_CMAKE_PATH ${_native_path} _native_path)
            DBG_MSG_V("Path ${_path} converted to ${_native_path}")
            string(APPEND _tmp_path " ${_native_path}")
          endif()
        endforeach()
        DBG_MSG("Setting wxWidgets_LIBRARY_DIRS = ${_tmp_path}")
        set(wxWidgets_LIBRARY_DIRS ${_tmp_path})
        separate_arguments(wxWidgets_LIBRARY_DIRS)
        list(REMOVE_ITEM wxWidgets_LIBRARY_DIRS "")
      endif()
      unset(_cygpath_exe CACHE)
    endif()

    # Check that all libraries are present, as wx-config does not check it
    set(_wx_lib_missing "")
    foreach(_wx_lib_ ${wxWidgets_LIBRARIES})
      if("${_wx_lib_}" MATCHES "^-l(.*)")
        set(_wx_lib_name "${CMAKE_MATCH_1}")
        if(_wx_lib_name STREQUAL "atomic")
          continue()
        endif()

        unset(_wx_lib_found CACHE)
        find_library(_wx_lib_found NAMES ${_wx_lib_name} HINTS ${wxWidgets_LIBRARY_DIRS})
        if(_wx_lib_found STREQUAL _wx_lib_found-NOTFOUND)
          list(APPEND _wx_lib_missing ${_wx_lib_name})
        endif()
        unset(_wx_lib_found CACHE)
      endif()
    endforeach()

    if (_wx_lib_missing)
      string(REPLACE ";" " " _wx_lib_missing "${_wx_lib_missing}")
      DBG_MSG_V("wxWidgets not found due to following missing libraries: ${_wx_lib_missing}")
      set(wxWidgets_FOUND FALSE)
      unset(wxWidgets_LIBRARIES)
    endif()
    unset(_wx_lib_missing)
endif()

# Check if a specific version was requested by find_package().
if(wxWidgets_FOUND)
  wx_extract_version()
endif()

file(TO_CMAKE_PATH "${wxWidgets_INCLUDE_DIRS}" wxWidgets_INCLUDE_DIRS)
file(TO_CMAKE_PATH "${wxWidgets_LIBRARY_DIRS}" wxWidgets_LIBRARY_DIRS)

# Debug output:
DBG_MSG("wxWidgets_FOUND           : ${wxWidgets_FOUND}")
DBG_MSG("wxWidgets_INCLUDE_DIRS    : ${wxWidgets_INCLUDE_DIRS}")
DBG_MSG("wxWidgets_LIBRARY_DIRS    : ${wxWidgets_LIBRARY_DIRS}")
DBG_MSG("wxWidgets_LIBRARIES       : ${wxWidgets_LIBRARIES}")
DBG_MSG("wxWidgets_CXX_FLAGS       : ${wxWidgets_CXX_FLAGS}")

#=====================================================================
#=====================================================================

include(FindPackageHandleStandardArgs)

# FIXME: set wxWidgets_<comp>_FOUND for wx-config branch
#        and use HANDLE_COMPONENTS on Unix too
if(wxWidgets_FIND_STYLE STREQUAL "win32")
  set(wxWidgets_HANDLE_COMPONENTS "HANDLE_COMPONENTS")
endif()

find_package_handle_standard_args(wxWidgets
  REQUIRED_VARS wxWidgets_LIBRARIES wxWidgets_INCLUDE_DIRS
  VERSION_VAR wxWidgets_VERSION
  ${wxWidgets_HANDLE_COMPONENTS}
  )
unset(wxWidgets_HANDLE_COMPONENTS)

if(wxWidgets_FOUND AND NOT TARGET wxWidgets::wxWidgets)
  add_library(wxWidgets::wxWidgets INTERFACE IMPORTED)
  target_link_libraries(wxWidgets::wxWidgets INTERFACE ${wxWidgets_LIBRARIES})
  target_link_directories(wxWidgets::wxWidgets INTERFACE ${wxWidgets_LIBRARY_DIRS})
  target_include_directories(wxWidgets::wxWidgets INTERFACE ${wxWidgets_INCLUDE_DIRS})
  target_compile_options(wxWidgets::wxWidgets INTERFACE ${wxWidgets_CXX_FLAGS})
  target_compile_definitions(wxWidgets::wxWidgets INTERFACE ${wxWidgets_DEFINITIONS})
  # FIXME: Add "$<$<CONFIG:Debug>:${wxWidgets_DEFINITIONS_DEBUG}>"
  # if the debug library variant is available.
endif()

#=====================================================================
# Macros for use in wxWidgets apps.
# - This module will not fail to find wxWidgets based on the code
#   below. Hence, it's required to check for validity of:
#
# wxWidgets_wxrc_EXECUTABLE
#=====================================================================

# Resource file compiler.
find_program(wxWidgets_wxrc_EXECUTABLE
  NAMES $ENV{WXRC_CMD} wxrc
  PATHS ${wxWidgets_ROOT_DIR}/utils/wxrc/vc_msw
  DOC "Location of wxWidgets resource file compiler binary (wxrc)"
  )

#
# WX_SPLIT_ARGUMENTS_ON(<keyword> <left> <right> <arg1> <arg2> ...)
#
# Sets <left> and <right> to contain arguments to the left and right,
# respectively, of <keyword>.
#
# Example usage:
#  function(WXWIDGETS_ADD_RESOURCES outfiles)
#    WX_SPLIT_ARGUMENTS_ON(OPTIONS wxrc_files wxrc_options ${ARGN})
#    ...
#  endfunction()
#
#  WXWIDGETS_ADD_RESOURCES(sources ${xrc_files} OPTIONS -e -o file.C)
#
# NOTE: This is a generic piece of code that should be renamed to
# SPLIT_ARGUMENTS_ON and put in a file serving the same purpose as
# FindPackageHandleStandardArgs.cmake. At the time of this writing
# FindQt4.cmake has a qt4_extract_options(), which I basically copied
# here a bit more generalized. So, there are already two find modules
# using this approach.
#
function(WX_SPLIT_ARGUMENTS_ON _keyword _leftvar _rightvar)
  # FIXME: Document that the input variables will be cleared.
  #list(APPEND ${_leftvar}  "")
  #list(APPEND ${_rightvar} "")
  set(${_leftvar}  "")
  set(${_rightvar} "")

  set(_doing_right FALSE)
  foreach(element ${ARGN})
    if("${element}" STREQUAL "${_keyword}")
      set(_doing_right TRUE)
    else()
      if(_doing_right)
        list(APPEND ${_rightvar} "${element}")
      else()
        list(APPEND ${_leftvar} "${element}")
      endif()
    endif()
  endforeach()

  set(${_leftvar}  ${${_leftvar}}  PARENT_SCOPE)
  set(${_rightvar} ${${_rightvar}} PARENT_SCOPE)
endfunction()

#
# WX_GET_DEPENDENCIES_FROM_XML(
#   <depends>
#   <match_pattern>
#   <clean_pattern>
#   <xml_contents>
#   <depends_path>
#   )
#
# FIXME: Add documentation here...
#
function(WX_GET_DEPENDENCIES_FROM_XML
    _depends
    _match_patt
    _clean_patt
    _xml_contents
    _depends_path
    )

  string(REGEX MATCHALL
    ${_match_patt}
    dep_file_list
    "${${_xml_contents}}"
    )
  foreach(dep_file ${dep_file_list})
    string(REGEX REPLACE ${_clean_patt} "" dep_file "${dep_file}")

    # make the file have an absolute path
    if(NOT IS_ABSOLUTE "${dep_file}")
      set(dep_file "${${_depends_path}}/${dep_file}")
    endif()

    # append file to dependency list
    list(APPEND ${_depends} "${dep_file}")
  endforeach()

  set(${_depends} ${${_depends}} PARENT_SCOPE)
endfunction()

#
# WXWIDGETS_ADD_RESOURCES(<sources> <xrc_files>
#                         OPTIONS <options> [NO_CPP_CODE])
#
# Adds a custom command for resource file compilation of the
# <xrc_files> and appends the output files to <sources>.
#
# Example usages:
#   WXWIDGETS_ADD_RESOURCES(sources xrc/main_frame.xrc)
#   WXWIDGETS_ADD_RESOURCES(sources ${xrc_files} OPTIONS -e -o altname.cxx)
#
function(WXWIDGETS_ADD_RESOURCES _outfiles)
  WX_SPLIT_ARGUMENTS_ON(OPTIONS rc_file_list rc_options ${ARGN})

  # Parse files for dependencies.
  set(rc_file_list_abs "")
  set(rc_depends       "")
  foreach(rc_file ${rc_file_list})
    get_filename_component(depends_path ${rc_file} PATH)

    get_filename_component(rc_file_abs ${rc_file} ABSOLUTE)
    list(APPEND rc_file_list_abs "${rc_file_abs}")

    # All files have absolute paths or paths relative to the location
    # of the rc file.
    file(READ "${rc_file_abs}" rc_file_contents)

    # get bitmap/bitmap2 files
    WX_GET_DEPENDENCIES_FROM_XML(
      rc_depends
      "<bitmap[^<]+"
      "^<bitmap[^>]*>"
      rc_file_contents
      depends_path
      )

    # get url files
    WX_GET_DEPENDENCIES_FROM_XML(
      rc_depends
      "<url[^<]+"
      "^<url[^>]*>"
      rc_file_contents
      depends_path
      )

    # get wxIcon files
    WX_GET_DEPENDENCIES_FROM_XML(
      rc_depends
      "<object[^>]*class=\"wxIcon\"[^<]+"
      "^<object[^>]*>"
      rc_file_contents
      depends_path
      )
  endforeach()

  #
  # Parse options.
  #
  # If NO_CPP_CODE option specified, then produce .xrs file rather
  # than a .cpp file (i.e., don't add the default --cpp-code option).
  list(FIND rc_options NO_CPP_CODE index)
  if(index EQUAL -1)
    list(APPEND rc_options --cpp-code)
    # wxrc's default output filename for cpp code.
    set(outfile resource.cpp)
  else()
    list(REMOVE_AT rc_options ${index})
    # wxrc's default output filename for xrs file.
    set(outfile resource.xrs)
  endif()

  # Get output name for use in ADD_CUSTOM_COMMAND.
  # - short option scanning
  list(FIND rc_options -o index)
  if(NOT index EQUAL -1)
    math(EXPR filename_index "${index} + 1")
    list(GET rc_options ${filename_index} outfile)
    #list(REMOVE_AT rc_options ${index} ${filename_index})
  endif()
  # - long option scanning
  string(REGEX MATCH "--output=[^;]*" outfile_opt "${rc_options}")
  if(outfile_opt)
    string(REPLACE "--output=" "" outfile "${outfile_opt}")
  endif()
  #string(REGEX REPLACE "--output=[^;]*;?" "" rc_options "${rc_options}")
  #string(REGEX REPLACE ";$" "" rc_options "${rc_options}")

  if(NOT IS_ABSOLUTE "${outfile}")
    set(outfile "${CMAKE_CURRENT_BINARY_DIR}/${outfile}")
  endif()
  add_custom_command(
    OUTPUT "${outfile}"
    COMMAND ${wxWidgets_wxrc_EXECUTABLE} ${rc_options} ${rc_file_list_abs}
    DEPENDS ${rc_file_list_abs} ${rc_depends}
    )

  # Add generated header to output file list.
  list(FIND rc_options -e short_index)
  list(FIND rc_options --extra-cpp-code long_index)
  if(NOT short_index EQUAL -1 OR NOT long_index EQUAL -1)
    get_filename_component(outfile_ext ${outfile} EXT)
    string(REPLACE "${outfile_ext}" ".h" outfile_header "${outfile}")
    list(APPEND ${_outfiles} "${outfile_header}")
    set_source_files_properties(
      "${outfile_header}" PROPERTIES GENERATED TRUE
      )
  endif()

  # Add generated file to output file list.
  list(APPEND ${_outfiles} "${outfile}")

  set(${_outfiles} ${${_outfiles}} PARENT_SCOPE)
endfunction()
