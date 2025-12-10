# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindKDE4
--------

.. note::

  This module is specifically intended for KDE version 4, which is obsolete
  and no longer maintained.  For modern application development using KDE
  technologies with CMake, use a newer version of KDE, and refer to the
  `KDE documentation
  <https://develop.kde.org/docs/getting-started/building/cmake-build/>`_.

Finds the KDE 4 installation:

.. code-block:: cmake

  find_package(KDE4 [...])

This module is a wrapper around the following upstream KDE 4 modules:

``FindKDE4Internal.cmake``

  Upstream internal module, which finds the KDE 4 include directories,
  libraries, and KDE-specific preprocessor tools.  It provides usage
  requirements for building KDE 4 software and defines several helper
  commands to simplify working with KDE 4 in CMake.

``KDE4Macros.cmake``
  Upstream utility module that defines all additional KDE4-specific
  commands to use KDE 4 in CMake.  For example:
  ``kde4_automoc()``, ``kde4_add_executable()``, ``kde4_add_library()``,
  ``kde4_add_ui_files()``, ``kde4_add_ui3_files()``,
  ``kde4_add_kcfg_files()``, ``kde4_add_kdeinit_executable()``, etc.

Upstream KDE 4 modules are installed by the KDE 4 distribution package in
``$KDEDIRS/share/apps/cmake/modules/``.  This path is automatically
appended to the :variable:`CMAKE_MODULE_PATH` variable when calling
``find_package(KDE4)``, so any additional KDE 4 modules can be included in
the project with :command:`include`.  For example:

``KDE4Defaults.cmake``
  Upstream internal module that sets some CMake options which are useful,
  but not required for building KDE 4 software.  If these settings should
  be used, include this module after finding KDE 4:

  .. code-block:: cmake

    find_package(KDE4)
    include(KDE4Defaults)

For usage details, refer to the upstream KDE 4 documentation.  For example,
at the top of the ``FindKDE4Internal`` module a complete documentation is
available for all variables and commands these modules provide.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``KDE4_FOUND``
  Boolean indicating whether KDE 4 was found.  This variable is set by the
  upstream ``FindKDE4Internal`` module.

Hints
^^^^^

This module accepts the following variables before calling the
``find_package(KDE4)``:

``ENV{KDEDIRS}``
  Environment variable containing the path to the KDE 4 installation.

KDE 4 is searched in the following directories in the given order:

* :variable:`CMAKE_INSTALL_PREFIX` variable
* ``KDEDIRS`` environment variable
* ``/opt/kde4`` path

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

Finding KDE 4 as required and using it in CMake:

.. code-block:: cmake

  find_package(KDE4 REQUIRED)

  set(sources main.cpp mywidget.cpp mypart.cpp)

  # The kde4_*() commands are provided by the KDE4Macros module, which is
  # included automatically by FindKDE4, if KDE4 is found:
  kde4_automoc(${sources})
  kde4_add_executable(example ${sources})

  target_include_directories(example PRIVATE ${KDE4_INCLUDES})
  target_link_libraries(example PRIVATE ${KDE4_KDEUI_LIBS} ${KDE4_KPARTS_LIBS})

  install(TARGETS example DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES kfoo.desktop DESTINATION ${XDG_APPS_DIR})

Example: Full Featured Example
""""""""""""""""""""""""""""""

In the following example this module is used to find KDE 4 installation.

.. code-block:: cmake

  project(kfoo)

  find_package(KDE4 REQUIRED)

  # Append path from where to include local project modules if any:
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  include_directories(${KDE4_INCLUDE_DIRS})
  add_definitions(${KDE4_DEFINITIONS})

  set(sources main.cpp myappl.cpp view.cpp)

  # If Qt designer UI files version 3 or 4 are available add them to the
  # sources variable:
  kde4_add_ui_files(sources maindialog.ui logindialog.ui)
  kde4_add_ui3_files(sources printerdlg.ui previewdlg.ui)

  # If there are files for the kconfig_compiler add them this way:
  kde4_add_kcfg_files(sources settings.kcfg)

  # When everything is listed, probably automoc is wanted:
  kde4_automoc(${sources})

  # Finally, specify what to build:
  kde4_add_executable(kfoo ${sources})


The ``kde4_add_executable()`` command is a slightly extended version of the
CMake command :command:`add_executable`.  Additionally, it does some more
``RPATH`` handling and supports the ``KDE4_ENABLE_FINAL`` variable.  The
first argument is the name of the executable followed by a list of source
files.  If a library needs to be created instead of an executable, the
``kde4_add_library()`` can be used.  It is an extended version of the
:command:`add_library` command.  It adds support for the
``KDE4_ENABLE_FINAL`` variable and under Windows it adds the
``-DMAKE_KFOO_LIB`` to the compile flags.

.. code-block:: cmake

  find_package(KDE4 REQUIRED)

  # ...

  kde4_add_library(kfoo ${sources})

  # Optionally, set the library version number if needed:
  set_target_properties(kfoo PROPERTIES VERSION 5.0.0 SOVERSION 5)

KDE is very modular, so if a KPart, a control center module, or an ioslave
needs to be created, here's how to do it:

.. code-block:: cmake

  find_package(KDE4 REQUIRED)
  # ...
  kde4_add_plugin(kfoo ${sources})

Now, the application/library/plugin probably needs to link to some
libraries.  For this use the standard :command:`target_link_libraries`
command.  For every KDE library there are variables available in the form
of ``KDE4_FOO_LIBS``.  Use them to get also all depending libraries:

.. code-block:: cmake

  target_link_libraries(kfoo ${KDE4_KDEUI_LIBS} ${KDE4_KIO_LIBS})

Example: The kdeinit Executable
"""""""""""""""""""""""""""""""

In the following example, the so called kdeinit executable is created.
The ``kde4_add_kdeinit_executable()`` command creates both an executable
with the given name and a library with the given name prefixed with
``kdeinit_``.  The :command:`target_link_libraries` command adds all
required libraries to the ``kdeinit_kbar`` library, and then links the
``kbar`` against the ``kdeinit_kbar``:

.. code-block:: cmake

  find_package(KDE4 REQUIRED)

  # ...

  kde4_add_kdeinit_executable(kbar ${kbarSources})
  target_link_libraries(kdeinit_kbar ${KDE4_KIO_LIBS})
  target_link_libraries(kbar kdeinit_kbar)

  install(TARGETS kbar DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(TARGETS kdeinit_kbar DESTINATION ${CMAKE_INSTALL_LIBDIR})

Example: Removing Compile Definitions
"""""""""""""""""""""""""""""""""""""

Sometimes, a default compile definition passed to the compiler needs to be
removed.  The :command:`remove_definitions` command can be used.  For
example, by default, the KDE4 build system sets the ``-DQT_NO_STL`` flag.
If the project code uses some of the Qt STL compatibility layer, this flag
should be removed:

.. code-block:: cmake

  find_package(KDE4 REQUIRED)

  add_definitions(${KDE4_DEFINITIONS})

  # ...

  remove_definitions(-DQT_NO_STL)
#]=======================================================================]

# Author: Alexander Neundorf <neundorf@kde.org>

# If Qt3 has already been found, fail.
if(QT_QT_LIBRARY)
  if(KDE4_FIND_REQUIRED)
    message( FATAL_ERROR "KDE4/Qt4 and Qt3 cannot be used together in one project.")
  else()
    if(NOT KDE4_FIND_QUIETLY)
      message( STATUS    "KDE4/Qt4 and Qt3 cannot be used together in one project.")
    endif()
    return()
  endif()
endif()

file(TO_CMAKE_PATH "$ENV{KDEDIRS}" _KDEDIRS)

# when cross compiling, searching kde4-config in order to run it later on
# doesn't make a lot of sense. We'll have to do something about this.
# Searching always in the target environment ? Then we get at least the correct one,
# still it can't be used to run it. Alex

# For KDE4 kde-config has been renamed to kde4-config
find_program(KDE4_KDECONFIG_EXECUTABLE NAMES kde4-config
   # the suffix must be used since KDEDIRS can be a list of directories which don't have bin/ appended
   PATH_SUFFIXES bin
   HINTS
   ${CMAKE_INSTALL_PREFIX}
   ${_KDEDIRS}
   /opt/kde4
   ONLY_CMAKE_FIND_ROOT_PATH
   )

if (NOT KDE4_KDECONFIG_EXECUTABLE)
  if (KDE4_FIND_REQUIRED)
    message(FATAL_ERROR "ERROR: Could not find KDE4 kde4-config")
  endif ()
endif ()


# when cross compiling, KDE4_DATA_DIR may be already preset
if(NOT KDE4_DATA_DIR)
  if(CMAKE_CROSSCOMPILING)
    # when cross compiling, don't run kde4-config but use its location as install dir
    get_filename_component(KDE4_DATA_DIR "${KDE4_KDECONFIG_EXECUTABLE}" PATH)
    get_filename_component(KDE4_DATA_DIR "${KDE4_DATA_DIR}" PATH)
  else()
    # then ask kde4-config for the kde data dirs

    if(KDE4_KDECONFIG_EXECUTABLE)
      execute_process(COMMAND "${KDE4_KDECONFIG_EXECUTABLE}" --path data OUTPUT_VARIABLE _data_DIR ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      file(TO_CMAKE_PATH "${_data_DIR}" _data_DIR)
      # then check the data dirs for FindKDE4Internal.cmake
      find_path(KDE4_DATA_DIR cmake/modules/FindKDE4Internal.cmake HINTS ${_data_DIR})
    endif()
  endif()
endif()

# if it has been found...
if (KDE4_DATA_DIR)

  set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${KDE4_DATA_DIR}/cmake/modules)

  if (KDE4_FIND_QUIETLY)
    set(_quiet QUIET)
  endif ()

  if (KDE4_FIND_REQUIRED)
    set(_req REQUIRED)
  endif ()

  # use FindKDE4Internal.cmake to do the rest
  find_package(KDE4Internal ${_req} ${_quiet} NO_POLICY_SCOPE)
else ()
  if (KDE4_FIND_REQUIRED)
    message(FATAL_ERROR "ERROR: cmake/modules/FindKDE4Internal.cmake not found in ${_data_DIR}")
  endif ()

  set(KDE4_FOUND FALSE)
endif ()
