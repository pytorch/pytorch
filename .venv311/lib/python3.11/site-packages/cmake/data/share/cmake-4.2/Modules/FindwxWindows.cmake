# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindwxWindows
-------------

.. deprecated:: 3.0

  Replaced by :module:`FindwxWidgets`.

Finds the wxWidgets (formerly known as wxWindows) installation and determines
the locations of its include directories and libraries, as well as the name of
the library:

.. code-block:: cmake

  find_package(wxWindows [...])

wxWidgets 2.6.x is supported for monolithic builds, such as those compiled in
the ``wx/build/msw`` directory using:

.. code-block:: shell

  nmake -f makefile.vc BUILD=debug SHARED=0 USE_OPENGL=1 MONOLITHIC=1

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``WXWINDOWS_FOUND``
  Boolean indicating whether the wxWidgets was found.
``WXWINDOWS_LIBRARIES``
  Libraries needed to link against to use wxWidgets.  This includes paths to
  the wxWidgets libraries and any additional linker flags, typically derived
  from the output of ``wx-config --libs`` on Unix/Linux systems.
``CMAKE_WXWINDOWS_CXX_FLAGS``
  Compiler options needed to use wxWidgets (if any).  On Linux, this corresponds
  to the output of ``wx-config --cxxflags``.
``WXWINDOWS_INCLUDE_DIR``
  The directory containing the ``wx/wx.h`` and ``wx/setup.h`` header files.
``WXWINDOWS_LINK_DIRECTORIES``
  Link directories, useful for setting ``rpath`` on Unix-like platforms.
``WXWINDOWS_DEFINITIONS``
  Extra compile definitions needed to use wxWidgets (if any).

Hints
^^^^^

This module accepts the following variables before calling the
``find_package(wxWindows)``:

``WXWINDOWS_USE_GL``
  Set this variable to boolean true to require OpenGL support.

``HAVE_ISYSTEM``
  Set this variable to boolean true to replace ``-I`` compiler options with
  ``-isystem`` when the C++ compiler is GNU (``g++``).

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``CMAKE_WX_CAN_COMPILE``
  .. deprecated:: 1.8
    Replaced by the ``WXWINDOWS_FOUND`` variable with the same value.

``WXWINDOWS_LIBRARY``
  .. deprecated:: 1.8
    Replaced by the ``WXWINDOWS_LIBRARIES`` variable with the same value.

``CMAKE_WX_CXX_FLAGS``
  .. deprecated:: 1.8
    Replaced by the ``CMAKE_WXWINDOWS_CXX_FLAGS`` variable with the same value.

``WXWINDOWS_INCLUDE_PATH``
  .. deprecated:: 1.8
    Replaced by the ``WXWINDOWS_INCLUDE_DIR`` variable with the same value.

Examples
^^^^^^^^

Example: Finding wxWidgets in Earlier CMake Versions
""""""""""""""""""""""""""""""""""""""""""""""""""""

In earlier versions of CMake, wxWidgets (wxWindows) could be found using:

.. code-block:: cmake

  find_package(wxWindows)

To request OpenGL support, the ``WXWINDOWS_USE_GL`` variable could be set before
calling ``find_package()``:

.. code-block:: cmake

  set(WXWINDOWS_USE_GL ON)
  find_package(wxWindows)

Using wxWidgets (wxWindows) in CMake was commonly done by including the
:module:`Use_wxWindows` module, which would find wxWidgets and set the
appropriate libraries, include directories, and compiler flags:

.. code-block:: cmake

  include(Use_wxWindows)

Example: Finding wxWidgets as of CMake 3.0
""""""""""""""""""""""""""""""""""""""""""

Starting with CMake 3.0, wxWidgets can be found using the
:module:`FindwxWidgets` module:

.. code-block:: cmake

  find_package(wxWidgets)
#]=======================================================================]

# AUTHOR Jan Woetzel (07/2003-01/2006)
# ------------------------------------------------------------------
#
# -removed OPTION for CMAKE_WXWINDOWS_USE_GL. Force the developer to SET it before calling this.
# -major update for wx 2.6.2 and monolithic build option. (10/2005)
#
# STATUS
# tested with:
#  cmake 1.6.7, Linux (Suse 7.3), wxWindows 2.4.0, gcc 2.95
#  cmake 1.6.7, Linux (Suse 8.2), wxWindows 2.4.0, gcc 3.3
#  cmake 1.6.7, Linux (Suse 8.2), wxWindows 2.4.1-patch1,  gcc 3.3
#  cmake 1.6.7, MS Windows XP home, wxWindows 2.4.1, MS Visual Studio .net 7 2002 (static build)
#  cmake 2.0.5 on Windows XP and Suse Linux 9.2
#  cmake 2.0.6 on Windows XP and Suse Linux 9.2, wxWidgets 2.6.2 MONOLITHIC build
#  cmake 2.2.2 on Windows XP, MS Visual Studio .net 2003 7.1 wxWidgets 2.6.2 MONOLITHIC build
#
# TODO
#  -OPTION for unicode builds
#  -further testing of DLL linking under MS WIN32
#  -better support for non-monolithic builds
#


if(WIN32)
  set(WIN32_STYLE_FIND 1)
endif()
if(MINGW)
  set(WIN32_STYLE_FIND 0)
  set(UNIX_STYLE_FIND 1)
endif()
if(UNIX)
  set(UNIX_STYLE_FIND 1)
endif()


if(WIN32_STYLE_FIND)

  ## ######################################################################
  ##
  ## Windows specific:
  ##
  ## candidates for root/base directory of wxwindows
  ## should have subdirs include and lib containing include/wx/wx.h
  ## fix the root dir to avoid mixing of headers/libs from different
  ## versions/builds:

  ## WX supports monolithic and multiple smaller libs (since 2.5.x), we prefer monolithic for now.
  ## monolithic = WX is built as a single big library
  ## e.g. compile on WIN32 as  "nmake -f makefile.vc MONOLITHIC=1 BUILD=debug SHARED=0 USE_OPENGL=1" (JW)
  option(WXWINDOWS_USE_MONOLITHIC "Use monolithic build of WX??" ON)
  mark_as_advanced(WXWINDOWS_USE_MONOLITHIC)

  ## GL libs used?
  option(WXWINDOWS_USE_GL "Use Wx with GL support(glcanvas)?" ON)
  mark_as_advanced(WXWINDOWS_USE_GL)


  ## avoid mixing of headers and libs between multiple installed WX versions,
  ## select just one tree here:
  find_path(WXWINDOWS_ROOT_DIR  include/wx/wx.h
    HINTS
      ENV WXWIN
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\wxWidgets_is1;Inno Setup: App Path]"  ## WX 2.6.x
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\wxWindows_is1;Inno Setup: App Path]"  ## WX 2.4.x
    PATHS
      C:/wxWidgets-2.6.2
      D:/wxWidgets-2.6.2
      C:/wxWidgets-2.6.1
      D:/wxWidgets-2.6.1
      C:/wxWindows-2.4.2
      D:/wxWindows-2.4.2
  )
  # message("DBG found WXWINDOWS_ROOT_DIR: ${WXWINDOWS_ROOT_DIR}")


  ## find libs for combination of static/shared with release/debug
  ## be careful if you add something here,
  ## avoid mixing of headers and libs of different wx versions,
  ## there may be multiple WX versions installed.
  set (WXWINDOWS_POSSIBLE_LIB_PATHS
    "${WXWINDOWS_ROOT_DIR}/lib"
    )

  ## monolithic?
  if (WXWINDOWS_USE_MONOLITHIC)

    find_library(WXWINDOWS_STATIC_LIBRARY
      NAMES wx wxmsw wxmsw26
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static release build library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY
      NAMES wxd wxmswd wxmsw26d
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug build library" )

    find_library(WXWINDOWS_SHARED_LIBRARY
      NAMES wxmsw26 wxmsw262 wxmsw24 wxmsw242 wxmsw241 wxmsw240 wx23_2 wx22_9
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_dll"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows shared release build library" )

    find_library(WXWINDOWS_SHARED_DEBUG_LIBRARY
      NAMES wxmsw26d wxmsw262d wxmsw24d wxmsw241d wxmsw240d wx23_2d wx22_9d
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_dll"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows shared debug build library " )


    ##
    ## required for WXWINDOWS_USE_GL
    ## gl lib is always build separate:
    ##
    find_library(WXWINDOWS_STATIC_LIBRARY_GL
      NAMES wx_gl wxmsw_gl wxmsw26_gl
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static release build GL library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_GL
      NAMES wxd_gl wxmswd_gl wxmsw26d_gl
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug build GL library" )


    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_PNG
      NAMES wxpngd
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug png library" )

    find_library(WXWINDOWS_STATIC_LIBRARY_PNG
      NAMES wxpng
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static png library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_TIFF
      NAMES wxtiffd
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug tiff library" )

    find_library(WXWINDOWS_STATIC_LIBRARY_TIFF
      NAMES wxtiff
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static tiff library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_JPEG
      NAMES wxjpegd  wxjpgd
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug jpeg library" )

    find_library(WXWINDOWS_STATIC_LIBRARY_JPEG
      NAMES wxjpeg wxjpg
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static jpeg library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_ZLIB
      NAMES wxzlibd
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug zlib library" )

    find_library(WXWINDOWS_STATIC_LIBRARY_ZLIB
      NAMES wxzlib
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static zib library" )

    find_library(WXWINDOWS_STATIC_DEBUG_LIBRARY_REGEX
      NAMES wxregexd
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static debug regex library" )

    find_library(WXWINDOWS_STATIC_LIBRARY_REGEX
      NAMES wxregex
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_lib"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static regex library" )



    ## untested:
    find_library(WXWINDOWS_SHARED_LIBRARY_GL
      NAMES wx_gl wxmsw_gl wxmsw26_gl
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_dll"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows shared release build GL library" )

    find_library(WXWINDOWS_SHARED_DEBUG_LIBRARY_GL
      NAMES wxd_gl wxmswd_gl wxmsw26d_gl
      PATHS
      "${WXWINDOWS_ROOT_DIR}/lib/vc_dll"
      ${WXWINDOWS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows shared debug build GL library" )


  else ()
    ## WX is built as multiple small pieces libraries instead of monolithic

    ## DEPRECATED (jw) replaced by more general WXWINDOWS_USE_MONOLITHIC ON/OFF
    # option(WXWINDOWS_SEPARATE_LIBS_BUILD "Is wxWindows build with separate libs?" OFF)

    ## HACK: This is very dirty.
    ## because the libs of a particular version are explicitly listed
    ## and NOT searched/verified.
    ## TODO:  Really search for each lib, then decide for
    ## monolithic x debug x shared x GL (=16 combinations) for at least 18 libs
    ## -->  about 288 combinations
    ## thus we need a different approach so solve this correctly ...

    message(STATUS "Warning: You are trying to use wxWidgets without monolithic build (WXWINDOWS_SEPARATE_LIBS_BUILD). This is a HACK, libraries are not verified! (JW).")

    set(WXWINDOWS_STATIC_LIBS ${WXWINDOWS_STATIC_LIBS}
      wxbase26
      wxbase26_net
      wxbase26_odbc
      wxbase26_xml
      wxmsw26_adv
      wxmsw26_core
      wxmsw26_dbgrid
      wxmsw26_gl
      wxmsw26_html
      wxmsw26_media
      wxmsw26_qa
      wxmsw26_xrc
      wxexpat
      wxjpeg
      wxpng
      wxregex
      wxtiff
      wxzlib
      comctl32
      rpcrt4
      wsock32
      )
    ## HACK: feed in to optimized / debug libraries if both were FOUND.
    set(WXWINDOWS_STATIC_DEBUG_LIBS ${WXWINDOWS_STATIC_DEBUG_LIBS}
      wxbase26d
      wxbase26d_net
      wxbase26d_odbc
      wxbase26d_xml
      wxmsw26d_adv
      wxmsw26d_core
      wxmsw26d_dbgrid
      wxmsw26d_gl
      wxmsw26d_html
      wxmsw26d_media
      wxmsw26d_qa
      wxmsw26d_xrc
      wxexpatd
      wxjpegd
      wxpngd
      wxregexd
      wxtiffd
      wxzlibd
      comctl32
      rpcrt4
      wsock32
      )
  endif ()


  ##
  ## now we should have found all WX libs available on the system.
  ## let the user decide which of the available onse to use.
  ##

  ## if there is at least one shared lib available
  ## let user choose whether to use shared or static wxwindows libs
  if(WXWINDOWS_SHARED_LIBRARY OR WXWINDOWS_SHARED_DEBUG_LIBRARY)
    ## default value OFF because wxWindows MSVS default build is static
    option(WXWINDOWS_USE_SHARED_LIBS
      "Use shared versions (dll) of wxWindows libraries?" OFF)
    mark_as_advanced(WXWINDOWS_USE_SHARED_LIBS)
  endif()

  ## add system libraries wxwindows always seems to depend on
  set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
    comctl32
    rpcrt4
    wsock32
    )

  if (NOT WXWINDOWS_USE_SHARED_LIBS)
    set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
      ##  these ones don't seem required, in particular  ctl3d32 is not necessary (Jan Woetzel 07/2003)
      #   ctl3d32
      debug ${WXWINDOWS_STATIC_DEBUG_LIBRARY_ZLIB}   optimized ${WXWINDOWS_STATIC_LIBRARY_ZLIB}
      debug ${WXWINDOWS_STATIC_DEBUG_LIBRARY_REGEX}  optimized ${WXWINDOWS_STATIC_LIBRARY_REGEX}
      debug ${WXWINDOWS_STATIC_DEBUG_LIBRARY_PNG}    optimized ${WXWINDOWS_STATIC_LIBRARY_PNG}
      debug ${WXWINDOWS_STATIC_DEBUG_LIBRARY_JPEG}   optimized ${WXWINDOWS_STATIC_LIBRARY_JPEG}
      debug ${WXWINDOWS_STATIC_DEBUG_LIBRARY_TIFF}   optimized ${WXWINDOWS_STATIC_LIBRARY_TIFF}
      )
  endif ()

  ## opengl/glu: TODO/FIXME: better use FindOpenGL.cmake here
  ## assume release versions of glu an dopengl, here.
  if (WXWINDOWS_USE_GL)
    set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
      opengl32
      glu32 )
  endif ()

  ##
  ## select between use of  shared or static wxWindows lib then set libs to use
  ## for debug and optimized build.  so the user can switch between debug and
  ## release build e.g. within MS Visual Studio without running cmake with a
  ## different build directory again.
  ##
  ## then add the build specific include dir for wx/setup.h
  ##

  if(WXWINDOWS_USE_SHARED_LIBS)
    ##message("DBG wxWindows use shared lib selected.")
    ## assume that both builds use the same setup(.h) for simplicity

    ## shared: both wx (debug and release) found?
    ## assume that both builds use the same setup(.h) for simplicity
    if(WXWINDOWS_SHARED_DEBUG_LIBRARY AND WXWINDOWS_SHARED_LIBRARY)
      ##message("DBG wx shared: debug and optimized found.")
      find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
        ${WXWINDOWS_ROOT_DIR}/lib/mswdlld
        ${WXWINDOWS_ROOT_DIR}/lib/mswdll
        ${WXWINDOWS_ROOT_DIR}/lib/vc_dll/mswd
        ${WXWINDOWS_ROOT_DIR}/lib/vc_dll/msw )
      set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
        debug     ${WXWINDOWS_SHARED_DEBUG_LIBRARY}
        optimized ${WXWINDOWS_SHARED_LIBRARY} )
      if (WXWINDOWS_USE_GL)
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          debug     ${WXWINDOWS_SHARED_DEBUG_LIBRARY_GL}
          optimized ${WXWINDOWS_SHARED_LIBRARY_GL} )
      endif ()
    endif()

    ## shared: only debug wx lib found?
    if(WXWINDOWS_SHARED_DEBUG_LIBRARY)
      if(NOT WXWINDOWS_SHARED_LIBRARY)
        ##message("DBG wx shared: debug (but no optimized) found.")
        find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWINDOWS_ROOT_DIR}/lib/mswdlld
          ${WXWINDOWS_ROOT_DIR}/lib/vc_dll/mswd  )
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          ${WXWINDOWS_SHARED_DEBUG_LIBRARY} )
        if (WXWINDOWS_USE_GL)
          set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
            ${WXWINDOWS_SHARED_DEBUG_LIBRARY_GL} )
        endif ()
      endif()
    endif()

    ## shared: only release wx lib found?
    if(NOT WXWINDOWS_SHARED_DEBUG_LIBRARY)
      if(WXWINDOWS_SHARED_LIBRARY)
        ##message("DBG wx shared: optimized (but no debug) found.")
        find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWINDOWS_ROOT_DIR}/lib/mswdll
          ${WXWINDOWS_ROOT_DIR}/lib/vc_dll/msw  )
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          ${WXWINDOWS_SHARED_DEBUG_LIBRARY} )
        if (WXWINDOWS_USE_GL)
          set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
            ${WXWINDOWS_SHARED_DEBUG_LIBRARY_GL} )
        endif ()
      endif()
    endif()

    ## shared: none found?
    if(NOT WXWINDOWS_SHARED_DEBUG_LIBRARY)
      if(NOT WXWINDOWS_SHARED_LIBRARY)
        message(STATUS
          "No shared wxWindows lib found, but WXWINDOWS_USE_SHARED_LIBS=${WXWINDOWS_USE_SHARED_LIBS}.")
      endif()
    endif()

    #########################################################################################
  else()

    ##jw: DEPRECATED if(NOT WXWINDOWS_SEPARATE_LIBS_BUILD)

    ## static: both wx (debug and release) found?
    ## assume that both builds use the same setup(.h) for simplicity
    if(WXWINDOWS_STATIC_DEBUG_LIBRARY AND WXWINDOWS_STATIC_LIBRARY)
      ##message("DBG wx static: debug and optimized found.")
      find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
        ${WXWINDOWS_ROOT_DIR}/lib/mswd
        ${WXWINDOWS_ROOT_DIR}/lib/msw
        ${WXWINDOWS_ROOT_DIR}/lib/vc_lib/mswd
        ${WXWINDOWS_ROOT_DIR}/lib/vc_lib/msw )
      set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
        debug     ${WXWINDOWS_STATIC_DEBUG_LIBRARY}
        optimized ${WXWINDOWS_STATIC_LIBRARY} )
      if (WXWINDOWS_USE_GL)
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          debug     ${WXWINDOWS_STATIC_DEBUG_LIBRARY_GL}
          optimized ${WXWINDOWS_STATIC_LIBRARY_GL} )
      endif ()
    endif()

    ## static: only debug wx lib found?
    if(WXWINDOWS_STATIC_DEBUG_LIBRARY)
      if(NOT WXWINDOWS_STATIC_LIBRARY)
        ##message("DBG wx static: debug (but no optimized) found.")
        find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWINDOWS_ROOT_DIR}/lib/mswd
          ${WXWINDOWS_ROOT_DIR}/lib/vc_lib/mswd  )
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          ${WXWINDOWS_STATIC_DEBUG_LIBRARY} )
        if (WXWINDOWS_USE_GL)
          set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
            ${WXWINDOWS_STATIC_DEBUG_LIBRARY_GL} )
        endif ()
      endif()
    endif()

    ## static: only release wx lib found?
    if(NOT WXWINDOWS_STATIC_DEBUG_LIBRARY)
      if(WXWINDOWS_STATIC_LIBRARY)
        ##message("DBG wx static: optimized (but no debug) found.")
        find_path(WXWINDOWS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWINDOWS_ROOT_DIR}/lib/msw
          ${WXWINDOWS_ROOT_DIR}/lib/vc_lib/msw )
        set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
          ${WXWINDOWS_STATIC_LIBRARY} )
        if (WXWINDOWS_USE_GL)
          set(WXWINDOWS_LIBRARIES ${WXWINDOWS_LIBRARIES}
            ${WXWINDOWS_STATIC_LIBRARY_GL} )
        endif ()
      endif()
    endif()

    ## static: none found?
    if(NOT WXWINDOWS_STATIC_DEBUG_LIBRARY AND NOT WXWINDOWS_SEPARATE_LIBS_BUILD)
      if(NOT WXWINDOWS_STATIC_LIBRARY)
        message(STATUS
          "No static wxWindows lib found, but WXWINDOWS_USE_SHARED_LIBS=${WXWINDOWS_USE_SHARED_LIBS}.")
      endif()
    endif()
  endif()


  ## not necessary in wxWindows 2.4.1 and 2.6.2
  ## but it may fix a previous bug, see
  ## http://lists.wxwindows.org/cgi-bin/ezmlm-cgi?8:mss:37574:200305:mpdioeneabobmgjenoap
  option(WXWINDOWS_SET_DEFINITIONS "Set additional defines for wxWindows" OFF)
  mark_as_advanced(WXWINDOWS_SET_DEFINITIONS)
  if (WXWINDOWS_SET_DEFINITIONS)
    set(WXWINDOWS_DEFINITIONS "-DWINVER=0x400")
  else ()
    # clear:
    set(WXWINDOWS_DEFINITIONS "")
  endif ()



  ## Find the include directories for wxwindows
  ## the first, build specific for wx/setup.h was determined before.
  ## add inc dir for general for "wx/wx.h"
  find_path(WXWINDOWS_INCLUDE_DIR  wx/wx.h
    "${WXWINDOWS_ROOT_DIR}/include" )
  ## append the build specific include dir for wx/setup.h:
  if (WXWINDOWS_INCLUDE_DIR_SETUPH)
    set(WXWINDOWS_INCLUDE_DIR ${WXWINDOWS_INCLUDE_DIR} ${WXWINDOWS_INCLUDE_DIR_SETUPH} )
  endif ()



  mark_as_advanced(
    WXWINDOWS_ROOT_DIR
    WXWINDOWS_INCLUDE_DIR
    WXWINDOWS_INCLUDE_DIR_SETUPH
    WXWINDOWS_STATIC_LIBRARY
    WXWINDOWS_STATIC_LIBRARY_GL
    WXWINDOWS_STATIC_DEBUG_LIBRARY
    WXWINDOWS_STATIC_DEBUG_LIBRARY_GL
    WXWINDOWS_STATIC_LIBRARY_ZLIB
    WXWINDOWS_STATIC_DEBUG_LIBRARY_ZLIB
    WXWINDOWS_STATIC_LIBRARY_REGEX
    WXWINDOWS_STATIC_DEBUG_LIBRARY_REGEX
    WXWINDOWS_STATIC_LIBRARY_PNG
    WXWINDOWS_STATIC_DEBUG_LIBRARY_PNG
    WXWINDOWS_STATIC_LIBRARY_JPEG
    WXWINDOWS_STATIC_DEBUG_LIBRARY_JPEG
    WXWINDOWS_STATIC_DEBUG_LIBRARY_TIFF
    WXWINDOWS_STATIC_LIBRARY_TIFF
    WXWINDOWS_SHARED_LIBRARY
    WXWINDOWS_SHARED_DEBUG_LIBRARY
    WXWINDOWS_SHARED_LIBRARY_GL
    WXWINDOWS_SHARED_DEBUG_LIBRARY_GL
    )


else()

  if (UNIX_STYLE_FIND)
    ## ######################################################################
    ##
    ## UNIX/Linux specific:
    ##
    ## use backquoted wx-config to query and set flags and libs:
    ## 06/2003 Jan Woetzel
    ##

    option(WXWINDOWS_USE_SHARED_LIBS "Use shared versions (.so) of wxWindows libraries" ON)
    mark_as_advanced(WXWINDOWS_USE_SHARED_LIBS)

    # JW removed option and force the developer to SET it.
    # option(WXWINDOWS_USE_GL "use wxWindows with GL support (use additional
    # --gl-libs for wx-config)?" OFF)

    # wx-config should be in your path anyhow, usually no need to set WXWIN or
    # search in ../wx or ../../wx
    find_program(CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE
      NAMES $ENV{WX_CONFIG} wx-config
      HINTS
        ENV WXWIN
        $ENV{WXWIN}/bin
      PATHS
      ../wx/bin
      ../../wx/bin )

    # check whether wx-config was found:
    if(CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE)

      # use shared/static wx lib?
      # remember: always link shared to use systems GL etc. libs (no static
      # linking, just link *against* static .a libs)
      if(WXWINDOWS_USE_SHARED_LIBS)
        set(WX_CONFIG_ARGS_LIBS --libs)
      else()
        set(WX_CONFIG_ARGS_LIBS --static --libs)
      endif()

      # do we need additionial wx GL stuff like GLCanvas ?
      if(WXWINDOWS_USE_GL)
        list(APPEND WX_CONFIG_ARGS_LIBS --gl-libs)
      endif()
      ##message("DBG: WX_CONFIG_ARGS_LIBS=${WX_CONFIG_ARGS_LIBS}===")

      # set CXXFLAGS to be fed into CMAKE_CXX_FLAGS by the user:
      if (HAVE_ISYSTEM) # does the compiler support -isystem ?
              if (NOT APPLE) # -isystem seems to be unsupported on Mac
                if(CMAKE_C_COMPILER_ID MATCHES "^(GNU|LCC)$" AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|LCC)$")
            if (CMAKE_CXX_COMPILER MATCHES g\\+\\+)
              set(CMAKE_WXWINDOWS_CXX_FLAGS "`${CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE} --cxxflags|sed -e s/-I/-isystem/g`")
            else()
              set(CMAKE_WXWINDOWS_CXX_FLAGS "`${CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE} --cxxflags`")
            endif()
                endif()
              endif ()
      endif ()
      ##message("DBG: for compilation:
      ##CMAKE_WXWINDOWS_CXX_FLAGS=${CMAKE_WXWINDOWS_CXX_FLAGS}===")

      # keep the back-quoted string for clarity
      string(REPLACE ";" " " _wx_config_args_libs "${WX_CONFIG_ARGS_LIBS}")
      set(WXWINDOWS_LIBRARIES "`${CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE} ${_wx_config_args_libs}`")
      ##message("DBG2: for linking:
      ##WXWINDOWS_LIBRARIES=${WXWINDOWS_LIBRARIES}===")

      # evaluate wx-config output to separate linker flags and linkdirs for
      # rpath:
      execute_process(COMMAND ${CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE}
        ${WX_CONFIG_ARGS_LIBS}
        OUTPUT_VARIABLE WX_CONFIG_LIBS )

      ## extract linkdirs (-L) for rpath
      ## use regular expression to match wildcard equivalent "-L*<endchar>"
      ## with <endchar> is a space or a semicolon
      string(REGEX MATCHALL "[-][L]([^ ;])+" WXWINDOWS_LINK_DIRECTORIES_WITH_PREFIX "${WX_CONFIG_LIBS}" )
      # message("DBG  WXWINDOWS_LINK_DIRECTORIES_WITH_PREFIX=${WXWINDOWS_LINK_DIRECTORIES_WITH_PREFIX}")

      ## remove prefix -L because we need the pure directory for LINK_DIRECTORIES
      ## replace -L by ; because the separator seems to be lost otherwise (bug or
      ## feature?)
      if(WXWINDOWS_LINK_DIRECTORIES_WITH_PREFIX)
        string(REGEX REPLACE "[-][L]" ";" WXWINDOWS_LINK_DIRECTORIES ${WXWINDOWS_LINK_DIRECTORIES_WITH_PREFIX} )
        # message("DBG  WXWINDOWS_LINK_DIRECTORIES=${WXWINDOWS_LINK_DIRECTORIES}")
      endif()


      ## replace space separated string by semicolon separated vector to make it
      ## work with LINK_DIRECTORIES
      separate_arguments(WXWINDOWS_LINK_DIRECTORIES)

      mark_as_advanced(
        CMAKE_WXWINDOWS_CXX_FLAGS
        WXWINDOWS_INCLUDE_DIR
        WXWINDOWS_LIBRARIES
        CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE
        )


      ## we really need wx-config...
    else()
      message(STATUS "Cannot find wx-config anywhere on the system. Please put the file into your path or specify it in CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE.")
      mark_as_advanced(CMAKE_WXWINDOWS_WXCONFIG_EXECUTABLE)
    endif()



  else()
    message(STATUS "FindwxWindows.cmake:  Platform unknown/unsupported by FindwxWindows.cmake. It's neither WIN32 nor UNIX")
  endif()
endif()


if(WXWINDOWS_LIBRARIES)
  if(WXWINDOWS_INCLUDE_DIR OR CMAKE_WXWINDOWS_CXX_FLAGS)
    ## found all we need.
    set(WXWINDOWS_FOUND 1)

    ## set deprecated variables for backward compatibility:
    set(CMAKE_WX_CAN_COMPILE   ${WXWINDOWS_FOUND})
    set(WXWINDOWS_LIBRARY     ${WXWINDOWS_LIBRARIES})
    set(WXWINDOWS_INCLUDE_PATH ${WXWINDOWS_INCLUDE_DIR})
    set(WXWINDOWS_LINK_DIRECTORIES ${WXWINDOWS_LINK_DIRECTORIES})
    set(CMAKE_WX_CXX_FLAGS     ${CMAKE_WXWINDOWS_CXX_FLAGS})

  endif()
endif()
