# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Qt4ConfigDependentSettings
--------------------------



This file is included by FindQt4.cmake, don't include it directly.
#]=======================================================================]

###############################################
#
#       configuration/system dependent settings
#
###############################################

# find dependencies for some Qt modules
# when doing builds against a static Qt, they are required
# when doing builds against a shared Qt, they are not required
# if a user needs the dependencies, and they couldn't be found, they can set
# the variables themselves.

set(QT_QTGUI_LIB_DEPENDENCIES "")
set(QT_QTCORE_LIB_DEPENDENCIES "")
set(QT_QTNETWORK_LIB_DEPENDENCIES "")
set(QT_QTOPENGL_LIB_DEPENDENCIES "")
set(QT_QTDBUS_LIB_DEPENDENCIES "")
set(QT_QTHELP_LIB_DEPENDENCIES ${QT_QTCLUCENE_LIBRARY})


if(Q_WS_WIN)
  # On Windows, qconfig.pri has "shared" for shared library builds
  if(NOT QT_CONFIG MATCHES "shared")
    set(QT_IS_STATIC 1)
  endif()
else()
  # On other platforms, check file extension to know if its static
  if(QT_QTCORE_LIBRARY_RELEASE)
    get_filename_component(qtcore_lib_ext "${QT_QTCORE_LIBRARY_RELEASE}" EXT)
    if("${qtcore_lib_ext}" STREQUAL "${CMAKE_STATIC_LIBRARY_SUFFIX}")
      set(QT_IS_STATIC 1)
    endif()
  endif()
  if(QT_QTCORE_LIBRARY_DEBUG)
    get_filename_component(qtcore_lib_ext "${QT_QTCORE_LIBRARY_DEBUG}" EXT)
    if(${qtcore_lib_ext} STREQUAL ${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(QT_IS_STATIC 1)
    endif()
  endif()
endif()

# build using shared Qt needs -DQT_DLL on Windows
if(Q_WS_WIN  AND  NOT QT_IS_STATIC)
  set(QT_DEFINITIONS ${QT_DEFINITIONS} -DQT_DLL)
endif()

if(NOT QT_IS_STATIC)
  return()
endif()

# QtOpenGL dependencies
find_package(OpenGL)
set (QT_QTOPENGL_LIB_DEPENDENCIES ${OPENGL_glu_LIBRARY} ${OPENGL_gl_LIBRARY})


## system png
if(QT_QCONFIG MATCHES "system-png")
  find_package(PNG)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${PNG_LIBRARY})
endif()

## system jpeg
if(QT_QCONFIG MATCHES "system-jpeg")
  find_package(JPEG)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${JPEG_LIBRARIES})
endif()

## system tiff
if(QT_QCONFIG MATCHES "system-tiff")
  find_package(TIFF)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${TIFF_LIBRARIES})
endif()

## system mng
if(QT_QCONFIG MATCHES "system-mng")
  find_library(MNG_LIBRARY NAMES mng)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${MNG_LIBRARY})
endif()

# for X11, get X11 library directory
if(Q_WS_X11)
  find_package(X11)
endif()


## X11 SM
if(QT_QCONFIG MATCHES "x11sm")
  if(X11_SM_LIB AND X11_ICE_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_SM_LIB} ${X11_ICE_LIB})
  endif()
endif()


## Xi
if(QT_QCONFIG MATCHES "tablet")
  if(X11_Xi_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xi_LIB})
  endif()
endif()


## Xrender
if(QT_QCONFIG MATCHES "xrender")
  if(X11_Xrender_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xrender_LIB})
  endif()
endif()


## Xrandr
if(QT_QCONFIG MATCHES "xrandr")
  if(X11_Xrandr_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xrandr_LIB})
  endif()
endif()


## Xcursor
if(QT_QCONFIG MATCHES "xcursor")
  if(X11_Xcursor_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xcursor_LIB})
  endif()
endif()


## Xinerama
if(QT_QCONFIG MATCHES "xinerama")
  if(X11_Xinerama_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xinerama_LIB})
  endif()
endif()


## Xfixes
if(QT_QCONFIG MATCHES "xfixes")
  if(X11_Xfixes_LIB)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xfixes_LIB})
  endif()
endif()


## fontconfig
if(QT_QCONFIG MATCHES "fontconfig")
  find_library(QT_FONTCONFIG_LIBRARY NAMES fontconfig)
  mark_as_advanced(QT_FONTCONFIG_LIBRARY)
  if(QT_FONTCONFIG_LIBRARY)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${QT_FONTCONFIG_LIBRARY})
  endif()
endif()


## system-freetype
if(QT_QCONFIG MATCHES "system-freetype")
  find_package(Freetype)
  if(FREETYPE_LIBRARIES)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${FREETYPE_LIBRARIES})
  endif()
endif()


## system-zlib
if(QT_QCONFIG MATCHES "system-zlib")
  find_package(ZLIB)
  set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ${ZLIB_LIBRARIES})
endif()


## openssl
if(NOT Q_WS_WIN)
  set(_QT_NEED_OPENSSL 0)
  if(QT_VERSION_MINOR LESS 4 AND QT_QCONFIG MATCHES "openssl")
    set(_QT_NEED_OPENSSL 1)
  endif()
  if(QT_VERSION_MINOR GREATER 3 AND QT_QCONFIG MATCHES "openssl-linked")
    set(_QT_NEED_OPENSSL 1)
  endif()
  if(_QT_NEED_OPENSSL)
    find_package(OpenSSL)
    if(OPENSSL_LIBRARIES)
      set(QT_QTNETWORK_LIB_DEPENDENCIES ${QT_QTNETWORK_LIB_DEPENDENCIES} ${OPENSSL_LIBRARIES})
    endif()
  endif()
endif()


## dbus
if(QT_QCONFIG MATCHES "dbus")

  find_library(QT_DBUS_LIBRARY NAMES dbus-1 )
  if(QT_DBUS_LIBRARY)
    set(QT_QTDBUS_LIB_DEPENDENCIES ${QT_QTDBUS_LIB_DEPENDENCIES} ${QT_DBUS_LIBRARY})
  endif()
  mark_as_advanced(QT_DBUS_LIBRARY)

endif()


## glib
if(QT_QCONFIG MATCHES "glib")

  # Qt 4.2.0+ uses glib-2.0
  find_library(QT_GLIB_LIBRARY NAMES glib-2.0 )
  find_library(QT_GTHREAD_LIBRARY NAMES gthread-2.0 )
  mark_as_advanced(QT_GLIB_LIBRARY)
  mark_as_advanced(QT_GTHREAD_LIBRARY)

  if(QT_GLIB_LIBRARY AND QT_GTHREAD_LIBRARY)
    set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES}
        ${QT_GTHREAD_LIBRARY} ${QT_GLIB_LIBRARY})
  endif()


  # Qt 4.5+ also links to gobject-2.0
  if(QT_VERSION_MINOR GREATER 4)
     find_library(QT_GOBJECT_LIBRARY NAMES gobject-2.0 PATHS ${_glib_query_output} )
     mark_as_advanced(QT_GOBJECT_LIBRARY)

     if(QT_GOBJECT_LIBRARY)
       set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES}
           ${QT_GOBJECT_LIBRARY})
     endif()
  endif()

endif()


## clock-monotonic, just see if we need to link with rt
if(QT_QCONFIG MATCHES "clock-monotonic")
  set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
  set(CMAKE_REQUIRED_LIBRARIES rt)
  check_symbol_exists(_POSIX_TIMERS "unistd.h;time.h" QT_POSIX_TIMERS)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
  if(QT_POSIX_TIMERS)
    find_library(QT_RT_LIBRARY NAMES rt)
    mark_as_advanced(QT_RT_LIBRARY)
    if(QT_RT_LIBRARY)
      set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ${QT_RT_LIBRARY})
    endif()
  endif()
endif()


if(Q_WS_X11)
  # X11 libraries Qt always depends on
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} ${X11_Xext_LIB} ${X11_X11_LIB})

  find_package(Threads)
  if(CMAKE_USE_PTHREADS_INIT)
    set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ${CMAKE_THREAD_LIBS_INIT})
  endif()

  set (QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ${CMAKE_DL_LIBS})

endif()


if(Q_WS_WIN)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} imm32 winmm)
  set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} ws2_32)
endif()


if(Q_WS_MAC)
  set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} "-framework Carbon")

  # Qt 4.0, 4.1, 4.2 use QuickTime
  if(QT_VERSION_MINOR LESS 3)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} "-framework QuickTime")
  endif()

  # Qt 4.2+ use AppKit
  if(QT_VERSION_MINOR GREATER 1)
    set(QT_QTGUI_LIB_DEPENDENCIES ${QT_QTGUI_LIB_DEPENDENCIES} "-framework AppKit")
  endif()

  set(QT_QTCORE_LIB_DEPENDENCIES ${QT_QTCORE_LIB_DEPENDENCIES} "-framework ApplicationServices")
endif()
