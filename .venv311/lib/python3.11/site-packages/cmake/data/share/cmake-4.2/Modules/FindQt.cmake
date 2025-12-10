# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindQt
------

.. deprecated:: 3.14

  This module is available only if policy :policy:`CMP0084` is not set to
  ``NEW``.  It supports only Qt3 and Qt4.  For Qt5 or later versions see
  :manual:`cmake-qt(7)`.

Finds an installed version of Qt3 or Qt4:

.. code-block:: cmake

  find_package(Qt [...])

Qt is a cross-platform application development framework for creating
graphical user interfaces and applications.

Use this module only if the project can work with both Qt3 and Qt4
versions.  If a specific version is required, use :module:`FindQt4` or
:module:`FindQt3` module directly.

Behavior:

* If multiple Qt versions are found, the user must set the preferred major
  Qt version with the ``DESIRED_QT_VERSION`` variable.
* If only one Qt version is found, then the ``DESIRED_QT_VERSION`` is set
  automatically.
* Once the ``DESIRED_QT_VERSION`` variable is set, the corresponding
  :module:`FindQt3` or :module:`FindQt4` module is included.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``QT4_INSTALLED``
  ``TRUE`` if Qt4 was found.
``QT3_INSTALLED``
  ``TRUE`` if Qt3 was found.

Hints
^^^^^

``QT_REQUIRED``
  If this variable is set to ``TRUE`` before calling ``find_package(Qt)``,
  CMake will raise an error if neither Qt3 nor Qt4 is found.
``DESIRED_QT_VERSION``
  Specifies the Qt major version to use.  Can be either ``3``, ``4``, or empty,
  to search for version automatically.

Examples
^^^^^^^^

Finding Qt3 or Qt4 version:

.. code-block:: cmake

  find_package(Qt)
#]=======================================================================]

if(_findqt_testing)
  set(_findqt_included TRUE)
  return()
endif()

# look for signs of qt3 installations
file(GLOB GLOB_TEMP_VAR /usr/lib*/qt-3*/bin/qmake /usr/lib*/qt3*/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT3_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

file(GLOB GLOB_TEMP_VAR /usr/local/qt-x11-commercial-3*/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT3_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

file(GLOB GLOB_TEMP_VAR /usr/local/lib/qt3/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT3_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

# look for qt4 installations
file(GLOB GLOB_TEMP_VAR /usr/local/qt-x11-commercial-4*/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT4_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

file(GLOB GLOB_TEMP_VAR /usr/local/Trolltech/Qt-4*/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT4_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

file(GLOB GLOB_TEMP_VAR /usr/local/lib/qt4/bin/qmake)
if(GLOB_TEMP_VAR)
  set(QT4_INSTALLED TRUE)
endif()
set(GLOB_TEMP_VAR)

if (Qt_FIND_VERSION)
  if (Qt_FIND_VERSION MATCHES "^([34])(\\.[0-9]+.*)?$")
    set(DESIRED_QT_VERSION ${CMAKE_MATCH_1})
  else ()
    message(FATAL_ERROR "FindQt was called with invalid version '${Qt_FIND_VERSION}'. Only Qt major versions 3 or 4 are supported. If you do not need to support both Qt3 and Qt4 in your source consider calling find_package(Qt3) or find_package(Qt4) instead of find_package(Qt) instead.")
  endif ()
endif ()

# now find qmake
find_program(QT_QMAKE_EXECUTABLE_FINDQT NAMES qmake PATHS "${QT_SEARCH_PATH}/bin" "$ENV{QTDIR}/bin")
if(QT_QMAKE_EXECUTABLE_FINDQT)
  execute_process(COMMAND ${QT_QMAKE_EXECUTABLE_FINDQT} -query QT_VERSION
    OUTPUT_VARIABLE QTVERSION)
  if(QTVERSION MATCHES "4")
    set(QT_QMAKE_EXECUTABLE ${QT_QMAKE_EXECUTABLE_FINDQT} CACHE PATH "Qt4 qmake program.")
    set(QT4_INSTALLED TRUE)
  endif()
  if(QTVERSION MATCHES "Unknown")
    set(QT3_INSTALLED TRUE)
  endif()
endif()

if(QT_QMAKE_EXECUTABLE_FINDQT)
  execute_process(COMMAND ${QT_QMAKE_EXECUTABLE_FINDQT}
    -query QT_INSTALL_HEADERS
    OUTPUT_VARIABLE qt_headers )
endif()

find_file( QT4_QGLOBAL_H_FILE qglobal.h
  "${QT_SEARCH_PATH}/Qt/include"
  "[HKEY_CURRENT_USER\\Software\\Trolltech\\Qt3Versions\\4.0.0;InstallDir]/include/Qt"
  "[HKEY_CURRENT_USER\\Software\\Trolltech\\Versions\\4.0.0;InstallDir]/include/Qt"
  ${qt_headers}/Qt
  $ENV{QTDIR}/include/Qt
  /usr/lib/qt/include/Qt
  /usr/share/qt4/include/Qt
  /usr/local/include/X11/qt4/Qt
  C:/Progra~1/qt/include/Qt
  PATH_SUFFIXES qt/include/Qt include/Qt)

if(QT4_QGLOBAL_H_FILE)
  set(QT4_INSTALLED TRUE)
endif()

find_file( QT3_QGLOBAL_H_FILE qglobal.h
  "${QT_SEARCH_PATH}/Qt/include"
 "[HKEY_CURRENT_USER\\Software\\Trolltech\\Qt3Versions\\3.2.1;InstallDir]/include/Qt"
  "[HKEY_CURRENT_USER\\Software\\Trolltech\\Qt3Versions\\3.2.0;InstallDir]/include/Qt"
  "[HKEY_CURRENT_USER\\Software\\Trolltech\\Qt3Versions\\3.1.0;InstallDir]/include/Qt"
  C:/Qt/3.3.3Educational/include
  $ENV{QTDIR}/include
  /usr/include/qt3/Qt
  /usr/share/qt3/include
  /usr/local/include/X11/qt3
  C:/Progra~1/qt/include
  PATH_SUFFIXES qt/include include/qt3)

if(QT3_QGLOBAL_H_FILE)
  set(QT3_INSTALLED TRUE)
endif()

if(QT3_INSTALLED AND QT4_INSTALLED AND NOT DESIRED_QT_VERSION)
  # force user to pick if we have both
  set(DESIRED_QT_VERSION 0 CACHE STRING "Pick a version of Qt to use: 3 or 4")
else()
  # if only one found then pick that one
  if(QT3_INSTALLED AND NOT DESIRED_QT_VERSION EQUAL 4)
    set(DESIRED_QT_VERSION 3 CACHE STRING "Pick a version of Qt to use: 3 or 4")
  endif()
  if(QT4_INSTALLED AND NOT DESIRED_QT_VERSION EQUAL 3)
    set(DESIRED_QT_VERSION 4 CACHE STRING "Pick a version of Qt to use: 3 or 4")
  endif()
endif()

if(DESIRED_QT_VERSION EQUAL 3)
  set(Qt3_FIND_REQUIRED ${Qt_FIND_REQUIRED})
  set(Qt3_FIND_QUIETLY  ${Qt_FIND_QUIETLY})
  include(${CMAKE_CURRENT_LIST_DIR}/FindQt3.cmake)
endif()
if(DESIRED_QT_VERSION EQUAL 4)
  set(Qt4_FIND_REQUIRED ${Qt_FIND_REQUIRED})
  set(Qt4_FIND_QUIETLY  ${Qt_FIND_QUIETLY})
  include(${CMAKE_CURRENT_LIST_DIR}/FindQt4.cmake)
endif()

if(NOT QT3_INSTALLED AND NOT QT4_INSTALLED)
  if(QT_REQUIRED)
    message(SEND_ERROR "CMake was unable to find any Qt versions, put qmake in your path, or set QT_QMAKE_EXECUTABLE.")
  endif()
else()
  if(NOT QT_FOUND AND NOT DESIRED_QT_VERSION)
    if(QT_REQUIRED)
      message(SEND_ERROR "Multiple versions of Qt found please set DESIRED_QT_VERSION")
    else()
      message("Multiple versions of Qt found please set DESIRED_QT_VERSION")
    endif()
  endif()
  if(NOT QT_FOUND AND DESIRED_QT_VERSION)
    if(QT_REQUIRED)
      message(FATAL_ERROR "CMake was unable to find Qt version: ${DESIRED_QT_VERSION}. Set advanced values QT_QMAKE_EXECUTABLE and QT${DESIRED_QT_VERSION}_QGLOBAL_H_FILE, if those are set then QT_QT_LIBRARY or QT_LIBRARY_DIR.")
    else()
      message( "CMake was unable to find desired Qt version: ${DESIRED_QT_VERSION}. Set advanced values QT_QMAKE_EXECUTABLE and QT${DESIRED_QT_VERSION}_QGLOBAL_H_FILE.")
    endif()
  endif()
endif()
mark_as_advanced(QT3_QGLOBAL_H_FILE QT4_QGLOBAL_H_FILE QT_QMAKE_EXECUTABLE_FINDQT)
