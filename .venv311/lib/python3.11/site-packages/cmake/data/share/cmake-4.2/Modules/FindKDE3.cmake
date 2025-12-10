# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindKDE3
--------

.. note::

  This module is specifically intended for KDE version 3, which is obsolete and
  no longer maintained.  For modern application development using KDE
  technologies with CMake, use a newer version of KDE, and refer to the
  `KDE documentation
  <https://develop.kde.org/docs/getting-started/building/cmake-build/>`_.

Finds KDE 3 include directories, libraries, and KDE-specific preprocessor
tools:

.. code-block:: cmake

  find_package(KDE3 [...])

This module provides usage requirements for building KDE 3 software and
defines several helper commands to simplify working with KDE 3 in CMake.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``KDE3_FOUND``
  Boolean indicating whether KDE 3 was found.
``KDE3_DEFINITIONS``
  Compiler definitions required for compiling KDE 3 software.
``KDE3_INCLUDE_DIRS``
  The KDE and the Qt include directories, for use with the
  :command:`target_include_directories` command.
``KDE3_LIB_DIR``
  The directory containing the installed KDE 3 libraries, for use with the
  :command:`target_link_directories` command.
``QT_AND_KDECORE_LIBS``
  A list containing both the Qt and the kdecore library, typically used together
  when linking KDE 3.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``KDE3_INCLUDE_DIR``
  The directory containing KDE 3 header files.
``KDE3_DCOPIDL_EXECUTABLE``
  The path to the ``dcopidl`` executable.
``KDE3_DCOPIDL2CPP_EXECUTABLE``
  The path to the ``dcopidl2cpp`` executable.
``KDE3_KCFGC_EXECUTABLE``
  The path to the ``kconfig_compiler`` executable.

Hints
^^^^^

This module accepts the following variables:

``KDE3_BUILD_TESTS``
  Provided as a user adjustable option.  Set this variable to boolean true to
  build KDE 3 testcases.

Commands
^^^^^^^^

This module provides the following commands to work with KDE 3 in CMake:

.. command:: kde3_automoc

  Enables automatic processing with ``moc`` for the given source files:

  .. code-block:: cmake

    kde3_automoc(<sources>...)

  Call this command to enable automatic ``moc`` file handling.  For example,
  if a source file (e.g., ``foo.cpp``) contains ``include "foo.moc"``, a
  ``moc`` file for the corresponding header (``foo.h``) will be generated
  automatically.  To skip processing for a specific source file, set the
  :prop_sf:`SKIP_AUTOMOC` source file property.

.. command:: kde3_add_moc_files

  Processes header files with ``moc``:

  .. code-block:: cmake

    kde3_add_moc_files(<variable> <headers>...)

  If not using ``kde3_automoc()``, this command can be used to generate ``moc``
  files for one or more ``<headers>`` files.  The generated files are named
  ``<filename>.moc.cpp`` and the resulting list of these generated source files
  is stored in the variable named ``<variable>`` for use in project targets.

.. command:: kde3_add_dcop_skels

  Generates KIDL and DCOP skeletons:

  .. code-block:: cmake

    kde3_add_dcop_skels(<variable> <dcop-headers>...)

  This command generates ``.kidl`` and DCOP skeleton source files from the given
  DCOP header files.  The resulting list of generated source files is stored in
  the variable named ``<variable>`` for use in project targets.

.. command:: kde3_add_dcop_stubs

  Generates DCOP stubs:

  .. code-block:: cmake

    kde3_add_dcop_stubs(<variable> <headers>...)

  Use this command to generate DCOP stubs from one or more given header files.
  The resulting list of generated source files is stored in the variable named
  ``<variable>`` for use in project targets.

.. command:: kde3_add_ui_files

  Adds Qt designer UI files:

  .. code-block:: cmake

    kde3_add_ui_files(<variable> <ui-files>...)

  This command creates the implementation files from the given Qt designer
  ``.ui`` files.  The resulting list of generated files is stored in the
  variable named ``<variable>`` for use in project targets.

.. command:: kde3_add_kcfg_files

  Adds KDE kconfig compiler files:

  .. code-block:: cmake

    kde3_add_kcfg_files(<variable> <kcfgc-files>...)

  Use this command to add KDE kconfig compiler files (``.kcfgc``) to the
  application/library.  The resulting list of generated source files is stored
  in the variable named ``<variable>`` for use in project targets.

.. command:: kde3_install_libtool_file

  Creates and installs a libtool file:

  .. code-block:: cmake

    kde3_install_libtool_file(<target>)

  This command creates and installs a basic libtool file for the given target
  ``<target>``.

.. command:: kde3_add_executable

  Adds KDE executable:

  .. code-block:: cmake

    kde3_add_executable(<name> <sources>...)

  This command is functionally identical to the built-in
  :command:`add_executable` command.  It was originally intended to support
  additional features in future versions of this module.

.. command:: kde3_add_kpart

  Creates a KDE plugin:

  .. code-block:: cmake

    kde3_add_kpart(<name> [WITH_PREFIX] <sources>...)

  This command creates a KDE plugin (KPart, kioslave, etc.) from one or more
  source files ``<sources>``.  It also creates and installs an appropriate
  libtool ``.la`` file.

  If the ``WITH_PREFIX`` option is given, the resulting plugin name will be
  prefixed with ``lib``.  Otherwise, no prefix is added.

.. command:: kde3_add_kdeinit_executable

  Creates a KDE application as a module loadable via kdeinit:

  .. code-block:: cmake

    kde3_add_kdeinit_executable(<name> <sources>...)

  This command creates a library named ``kdeinit_<name>`` from one or more
  source files ``<sources>``.  It also builds a small executable linked against
  this library.

Examples
^^^^^^^^

Finding KDE 3:

.. code-block:: cmake

  find_package(KDE3)
#]=======================================================================]

# Author: Alexander Neundorf <neundorf@kde.org>

if(NOT UNIX AND KDE3_FIND_REQUIRED)
  message(FATAL_ERROR "Compiling KDE3 applications and libraries under Windows is not supported")
endif()

# If Qt4 has already been found, fail.
if(Qt4_FOUND)
  if(KDE3_FIND_REQUIRED)
    message( FATAL_ERROR "KDE3/Qt3 and Qt4 cannot be used together in one project.")
  else()
    if(NOT KDE3_FIND_QUIETLY)
      message( STATUS    "KDE3/Qt3 and Qt4 cannot be used together in one project.")
    endif()
    return()
  endif()
endif()


set(QT_MT_REQUIRED TRUE)
#set(QT_MIN_VERSION "3.0.0")

#this line includes FindQt.cmake, which searches the Qt library and headers
if(KDE3_FIND_REQUIRED)
  set(_REQ_STRING_KDE3 "REQUIRED")
endif()

find_package(Qt3 ${_REQ_STRING_KDE3})
find_package(X11 ${_REQ_STRING_KDE3})


#now try to find some kde stuff
find_program(KDECONFIG_EXECUTABLE NAMES kde-config
  HINTS
    $ENV{KDEDIR}/bin
  PATHS
    /opt/kde3/bin
    /opt/kde/bin
  )

set(KDE3PREFIX)
if(KDECONFIG_EXECUTABLE)
  execute_process(COMMAND ${KDECONFIG_EXECUTABLE} --version
                  OUTPUT_VARIABLE kde_config_version )

  string(REGEX MATCH "KDE: .\\." kde_version "${kde_config_version}")
  if ("${kde_version}" MATCHES "KDE: 3\\.")
    execute_process(COMMAND ${KDECONFIG_EXECUTABLE} --prefix
                    OUTPUT_VARIABLE kdedir )
    string(REPLACE "\n" "" KDE3PREFIX "${kdedir}")

  endif ()
endif()



# at first the KDE include directory
# kpassdlg.h comes from kdeui and doesn't exist in KDE4 anymore
find_path(KDE3_INCLUDE_DIR kpassdlg.h
  HINTS
    $ENV{KDEDIR}/include
    ${KDE3PREFIX}/include
  PATHS
    /opt/kde3/include
    /opt/kde/include
  PATH_SUFFIXES include/kde
  )

#now the KDE library directory
find_library(KDE3_KDECORE_LIBRARY NAMES kdecore
  HINTS
    $ENV{KDEDIR}/lib
    ${KDE3PREFIX}/lib
  PATHS
    /opt/kde3/lib
    /opt/kde/lib
)

set(QT_AND_KDECORE_LIBS ${QT_LIBRARIES} ${KDE3_KDECORE_LIBRARY})

get_filename_component(KDE3_LIB_DIR ${KDE3_KDECORE_LIBRARY} PATH )

if(NOT KDE3_LIBTOOL_DIR)
  if(KDE3_KDECORE_LIBRARY MATCHES lib64)
    set(KDE3_LIBTOOL_DIR /lib64/kde3)
  elseif(KDE3_KDECORE_LIBRARY MATCHES libx32)
    set(KDE3_LIBTOOL_DIR /libx32/kde3)
  else()
    set(KDE3_LIBTOOL_DIR /lib/kde3)
  endif()
endif()

#now search for the dcop utilities
find_program(KDE3_DCOPIDL_EXECUTABLE NAMES dcopidl
  HINTS
    $ENV{KDEDIR}/bin
    ${KDE3PREFIX}/bin
  PATHS
    /opt/kde3/bin
    /opt/kde/bin
  )

find_program(KDE3_DCOPIDL2CPP_EXECUTABLE NAMES dcopidl2cpp
  HINTS
    $ENV{KDEDIR}/bin
    ${KDE3PREFIX}/bin
  PATHS
    /opt/kde3/bin
    /opt/kde/bin
  )

find_program(KDE3_KCFGC_EXECUTABLE NAMES kconfig_compiler
  HINTS
    $ENV{KDEDIR}/bin
    ${KDE3PREFIX}/bin
  PATHS
    /opt/kde3/bin
    /opt/kde/bin
  )


#SET KDE3_FOUND
if (KDE3_INCLUDE_DIR AND KDE3_LIB_DIR AND KDE3_DCOPIDL_EXECUTABLE AND KDE3_DCOPIDL2CPP_EXECUTABLE AND KDE3_KCFGC_EXECUTABLE)
  set(KDE3_FOUND TRUE)
else ()
  set(KDE3_FOUND FALSE)
endif ()

# add some KDE specific stuff
set(KDE3_DEFINITIONS -DQT_CLEAN_NAMESPACE -D_GNU_SOURCE)

# set compiler flags only if KDE3 has actually been found
if(KDE3_FOUND)
  set(_KDE3_USE_FLAGS FALSE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|LCC)$")
    set(_KDE3_USE_FLAGS TRUE) # use flags for gnu compiler
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                    OUTPUT_VARIABLE out)
    # gnu gcc 2.96 does not work with flags
    # I guess 2.95 also doesn't then
    if("${out}" MATCHES "2.9[56]")
      set(_KDE3_USE_FLAGS FALSE)
    endif()
  endif()

  #only on linux, but NOT e.g. on FreeBSD:
  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND _KDE3_USE_FLAGS)
    set (KDE3_DEFINITIONS ${KDE3_DEFINITIONS} -D_XOPEN_SOURCE=500 -D_BSD_SOURCE)
    set ( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -Wno-long-long -ansi -Wundef -Wcast-align -Wconversion -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -Wmissing-format-attribute -fno-common")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -ansi -Wundef -Wcast-align -Wconversion -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -fno-exceptions -fno-check-new -fno-common")
  endif()

  # works on FreeBSD, NOT tested on NetBSD and OpenBSD
  if (CMAKE_SYSTEM_NAME MATCHES BSD AND _KDE3_USE_FLAGS)
    set ( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -Wno-long-long -ansi -Wundef -Wcast-align -Wconversion -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -Wmissing-format-attribute -fno-common")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-long-long -Wundef -Wcast-align -Wconversion -Wchar-subscripts -Wall -W -Wpointer-arith -Wwrite-strings -Wformat-security -Wmissing-format-attribute -fno-exceptions -fno-check-new -fno-common")
  endif ()

  # if no special buildtype is selected, add -O2 as default optimization
  if (NOT CMAKE_BUILD_TYPE AND _KDE3_USE_FLAGS)
    set ( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -O2")
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  endif ()

#set(CMAKE_SHARED_LINKER_FLAGS "-avoid-version -module -Wl,--no-undefined -Wl,--allow-shlib-undefined")
#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--fatal-warnings -avoid-version -Wl,--no-undefined -lc")
#set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--fatal-warnings -avoid-version -Wl,--no-undefined -lc")
endif()


# KDE3Macros.cmake contains all the KDE specific macros
include(${CMAKE_CURRENT_LIST_DIR}/KDE3Macros.cmake)


macro (KDE3_PRINT_RESULTS)
  if(KDE3_INCLUDE_DIR)
    message(STATUS "Found KDE3 include dir: ${KDE3_INCLUDE_DIR}")
  else()
    message(STATUS "Didn't find KDE3 headers")
  endif()

  if(KDE3_LIB_DIR)
    message(STATUS "Found KDE3 library dir: ${KDE3_LIB_DIR}")
  else()
    message(STATUS "Didn't find KDE3 core library")
  endif()

  if(KDE3_DCOPIDL_EXECUTABLE)
    message(STATUS "Found KDE3 dcopidl preprocessor: ${KDE3_DCOPIDL_EXECUTABLE}")
  else()
    message(STATUS "Didn't find the KDE3 dcopidl preprocessor")
  endif()

  if(KDE3_DCOPIDL2CPP_EXECUTABLE)
    message(STATUS "Found KDE3 dcopidl2cpp preprocessor: ${KDE3_DCOPIDL2CPP_EXECUTABLE}")
  else()
    message(STATUS "Didn't find the KDE3 dcopidl2cpp preprocessor")
  endif()

  if(KDE3_KCFGC_EXECUTABLE)
    message(STATUS "Found KDE3 kconfig_compiler preprocessor: ${KDE3_KCFGC_EXECUTABLE}")
  else()
    message(STATUS "Didn't find the KDE3 kconfig_compiler preprocessor")
  endif()

endmacro ()


if (KDE3_FIND_REQUIRED AND NOT KDE3_FOUND)
  #bail out if something wasn't found
  KDE3_PRINT_RESULTS()
  message(FATAL_ERROR "Could NOT find everything required for compiling KDE 3 programs")

endif ()


if (NOT KDE3_FIND_QUIETLY)
  KDE3_PRINT_RESULTS()
endif ()

#add the found Qt and KDE include directories to the current include path
set(KDE3_INCLUDE_DIRS ${QT_INCLUDE_DIR} ${KDE3_INCLUDE_DIR})
