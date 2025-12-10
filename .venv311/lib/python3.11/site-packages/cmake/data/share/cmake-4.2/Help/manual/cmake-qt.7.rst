.. cmake-manual-description: CMake Qt Features Reference

cmake-qt(7)
***********

.. only:: html

   .. contents::

Introduction
============

CMake can find and use Qt 4, Qt 5 and Qt 6 libraries. The Qt 4 libraries are
found by the :module:`FindQt4` find-module shipped with CMake, whereas the
Qt 5 and Qt 6 libraries are found using "Config-file Packages" shipped with
Qt 5 and Qt 6. See :manual:`cmake-packages(7)` for more information about CMake
packages, and see `the Qt cmake manual`_ for your Qt version.

.. _`the Qt cmake manual`: https://doc.qt.io/qt-6/cmake-manual.html

Qt 4, Qt 5 and Qt 6 may be used together in the same
:manual:`CMake buildsystem <cmake-buildsystem(7)>`:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

  project(Qt4_5_6)

  set(CMAKE_AUTOMOC ON)

  find_package(Qt6 COMPONENTS Widgets DBus REQUIRED)
  add_executable(publisher publisher.cpp)
  target_link_libraries(publisher Qt6::Widgets Qt6::DBus)

  find_package(Qt5 COMPONENTS Gui DBus REQUIRED)
  add_executable(subscriber1 subscriber1.cpp)
  target_link_libraries(subscriber1 Qt5::Gui Qt5::DBus)

  find_package(Qt4 REQUIRED)
  add_executable(subscriber2 subscriber2.cpp)
  target_link_libraries(subscriber2 Qt4::QtGui Qt4::QtDBus)

A CMake target may not link to more than one Qt version.  A diagnostic is issued
if this is attempted or results from transitive target dependency evaluation.

Qt Build Tools
==============

Qt relies on some bundled tools for code generation, such as ``moc`` for
meta-object code generation, ``uic`` for widget layout and population,
and ``rcc`` for virtual file system content generation.  These tools may be
automatically invoked by :manual:`cmake(1)` if the appropriate conditions
are met.  The automatic tool invocation may be used with Qt version 4 to 6.

.. _`Qt AUTOMOC`:

AUTOMOC
^^^^^^^

The :prop_tgt:`AUTOMOC` target property controls whether :manual:`cmake(1)`
inspects the C++ files in the target to determine if they require ``moc`` to
be run, and to create rules to execute ``moc`` at the appropriate time.

If a macro from :prop_tgt:`AUTOMOC_MACRO_NAMES` is found in a header file,
``moc`` will be run on the file.  The result will be put into a file named
according to ``moc_<basename>.cpp``.
If the macro is found in a C++ implementation
file, the moc output will be put into a file named according to
``<basename>.moc``, following the Qt conventions.  The ``<basename>.moc`` must
be included by the user in the C++ implementation file with a preprocessor
``#include``.

Included ``moc_*.cpp`` and ``*.moc`` files will be generated in the
``<AUTOGEN_BUILD_DIR>/include`` directory which is
automatically added to the target's :prop_tgt:`INCLUDE_DIRECTORIES`.

* This differs from CMake 3.7 and below; see their documentation for details.

* For :prop_gbl:`multi configuration generators <GENERATOR_IS_MULTI_CONFIG>`,
  the include directory is ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>``.

* See :prop_tgt:`AUTOGEN_BUILD_DIR`.

Not included ``moc_<basename>.cpp`` files will be generated in custom
folders to avoid name collisions and included in a separate
file which is compiled into the target, named either
``<AUTOGEN_BUILD_DIR>/mocs_compilation.cpp`` or
``<AUTOGEN_BUILD_DIR>/mocs_compilation_$<CONFIG>.cpp``.

* See :prop_tgt:`AUTOGEN_BUILD_DIR`.

The ``moc`` command line will consume the :prop_tgt:`COMPILE_DEFINITIONS` and
:prop_tgt:`INCLUDE_DIRECTORIES` target properties from the target it is being
invoked for, and for the appropriate build configuration.

The :prop_tgt:`AUTOMOC` target property may be pre-set for all
following targets by setting the :variable:`CMAKE_AUTOMOC` variable.  The
:prop_tgt:`AUTOMOC_MOC_OPTIONS` target property may be populated to set
options to pass to ``moc``. The :variable:`CMAKE_AUTOMOC_MOC_OPTIONS`
variable may be populated to pre-set the options for all following targets.

Additional macro names to search for can be added to
:prop_tgt:`AUTOMOC_MACRO_NAMES`.

Additional ``moc`` dependency file names can be extracted from source code
by using :prop_tgt:`AUTOMOC_DEPEND_FILTERS`.

Source C++ files can be excluded from :prop_tgt:`AUTOMOC` processing by
enabling :prop_sf:`SKIP_AUTOMOC` or the broader :prop_sf:`SKIP_AUTOGEN`.

.. _`Qt AUTOUIC`:

AUTOUIC
^^^^^^^

The :prop_tgt:`AUTOUIC` target property controls whether :manual:`cmake(1)`
inspects the C++ files in the target to determine if they require ``uic`` to
be run, and to create rules to execute ``uic`` at the appropriate time.

If a preprocessor ``#include`` directive is found which matches
``<path>ui_<basename>.h``, and a ``<basename>.ui`` file exists,
then ``uic`` will be executed to generate the appropriate file.
The ``<basename>.ui`` file is searched for in the following places

1. ``<source_dir>/<basename>.ui``
2. ``<source_dir>/<path><basename>.ui``
3. ``<AUTOUIC_SEARCH_PATHS>/<basename>.ui``
4. ``<AUTOUIC_SEARCH_PATHS>/<path><basename>.ui``

where ``<source_dir>`` is the directory of the C++ file and
:prop_tgt:`AUTOUIC_SEARCH_PATHS` is a list of additional search paths.

The generated generated ``ui_*.h`` files are placed in the
``<AUTOGEN_BUILD_DIR>/include`` directory which is
automatically added to the target's :prop_tgt:`INCLUDE_DIRECTORIES`.

* This differs from CMake 3.7 and below; see their documentation for details.

* For :prop_gbl:`multi configuration generators <GENERATOR_IS_MULTI_CONFIG>`,
  the include directory is ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>``.

* See :prop_tgt:`AUTOGEN_BUILD_DIR`.

The :prop_tgt:`AUTOUIC` target property may be pre-set for all following
targets by setting the :variable:`CMAKE_AUTOUIC` variable.  The
:prop_tgt:`AUTOUIC_OPTIONS` target property may be populated to set options
to pass to ``uic``.  The :variable:`CMAKE_AUTOUIC_OPTIONS` variable may be
populated to pre-set the options for all following targets.  The
:prop_sf:`AUTOUIC_OPTIONS` source file property may be set on the
``<basename>.ui`` file to set particular options for the file.  This
overrides options from the :prop_tgt:`AUTOUIC_OPTIONS` target property.

A target may populate the :prop_tgt:`INTERFACE_AUTOUIC_OPTIONS` target
property with options that should be used when invoking ``uic``.  This must be
consistent with the :prop_tgt:`AUTOUIC_OPTIONS` target property content of the
depender target.  The :variable:`CMAKE_DEBUG_TARGET_PROPERTIES` variable may
be used to track the origin target of such
:prop_tgt:`INTERFACE_AUTOUIC_OPTIONS`.  This means that a library which
provides an alternative translation system for Qt may specify options which
should be used when running ``uic``:

.. code-block:: cmake

  add_library(KI18n klocalizedstring.cpp)
  target_link_libraries(KI18n Qt6::Core)

  # KI18n uses the tr2i18n() function instead of tr().  That function is
  # declared in the klocalizedstring.h header.
  set(autouic_options
    -tr tr2i18n
    -include klocalizedstring.h
  )

  set_property(TARGET KI18n APPEND PROPERTY
    INTERFACE_AUTOUIC_OPTIONS ${autouic_options}
  )

A consuming project linking to the target exported from upstream automatically
uses appropriate options when ``uic`` is run by :prop_tgt:`AUTOUIC`, as a
result of linking with the :prop_tgt:`IMPORTED` target:

.. code-block:: cmake

  set(CMAKE_AUTOUIC ON)
  # Uses a libwidget.ui file:
  add_library(LibWidget libwidget.cpp)
  target_link_libraries(LibWidget
    KF5::KI18n
    Qt5::Widgets
  )

Source files can be excluded from :prop_tgt:`AUTOUIC` processing by
enabling :prop_sf:`SKIP_AUTOUIC` or the broader :prop_sf:`SKIP_AUTOGEN`.

.. _`Qt AUTORCC`:

AUTORCC
^^^^^^^

The :prop_tgt:`AUTORCC` target property controls whether :manual:`cmake(1)`
creates rules to execute ``rcc`` at the appropriate time on source files
which have the suffix ``.qrc``.

.. code-block:: cmake

  add_executable(myexe main.cpp resource_file.qrc)

The :prop_tgt:`AUTORCC` target property may be pre-set for all following targets
by setting the :variable:`CMAKE_AUTORCC` variable.  The
:prop_tgt:`AUTORCC_OPTIONS` target property may be populated to set options
to pass to ``rcc``.  The :variable:`CMAKE_AUTORCC_OPTIONS` variable may be
populated to pre-set the options for all following targets.  The
:prop_sf:`AUTORCC_OPTIONS` source file property may be set on the
``<name>.qrc`` file to set particular options for the file.  This
overrides options from the :prop_tgt:`AUTORCC_OPTIONS` target property.

Source files can be excluded from :prop_tgt:`AUTORCC` processing by
enabling :prop_sf:`SKIP_AUTORCC` or the broader :prop_sf:`SKIP_AUTOGEN`.

.. _`<ORIGIN>_autogen`:

The ``<ORIGIN>_autogen`` target
===============================

The ``moc`` and ``uic`` tools are executed as part of a synthesized
``<ORIGIN>_autogen`` :command:`custom target <add_custom_target>` generated by
CMake.  By default, that ``<ORIGIN>_autogen`` target inherits the dependencies
of the ``<ORIGIN>`` target (see :prop_tgt:`AUTOGEN_ORIGIN_DEPENDS`).
Target dependencies may be added to the ``<ORIGIN>_autogen`` target by adding
them to the :prop_tgt:`AUTOGEN_TARGET_DEPENDS` target property.

.. note::
  If Qt 5.15 or later is used and the generator is either :generator:`Ninja` or
  :ref:`Makefile Generators`, see :ref:`<ORIGIN>_autogen_timestamp_deps`.

.. _`<ORIGIN>_autogen_timestamp_deps`:

The ``<ORIGIN>_autogen_timestamp_deps`` target
==============================================

If Qt 5.15 or later is used and the generator is either :generator:`Ninja` or
:ref:`Makefile Generators`, the ``<ORIGIN>_autogen_timestamp_deps`` target is
also created in addition to the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
target.  This target does not have any sources or commands to execute, but it
has dependencies that were previously inherited by the pre-Qt 5.15
:ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target.
These dependencies will serve as a list of order-only dependencies for the
custom command, without forcing the custom command to re-execute.

Visual Studio Generators
========================

When using the :ref:`Visual Studio Generators`, CMake
generates a ``PRE_BUILD`` :command:`custom command <add_custom_command>`
instead of the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
:command:`custom target <add_custom_target>` (for :prop_tgt:`AUTOMOC` and
:prop_tgt:`AUTOUIC`).  This isn't always possible though and an
:ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
:command:`custom target <add_custom_target>` is used, when either

- the ``<ORIGIN>`` target depends on :prop_sf:`GENERATED` files which aren't
  excluded from :prop_tgt:`AUTOMOC` and :prop_tgt:`AUTOUIC` by
  :prop_sf:`SKIP_AUTOMOC`, :prop_sf:`SKIP_AUTOUIC`, :prop_sf:`SKIP_AUTOGEN`
  or :policy:`CMP0071`
- :prop_tgt:`AUTOGEN_TARGET_DEPENDS` lists a source file
- :variable:`CMAKE_GLOBAL_AUTOGEN_TARGET` is enabled

qtmain.lib on Windows
=====================

The Qt 4 and 5 :prop_tgt:`IMPORTED` targets for the QtGui libraries specify
that the qtmain.lib static library shipped with Qt will be linked by all
dependent executables which have the :prop_tgt:`WIN32_EXECUTABLE` enabled.

To disable this behavior, enable the ``Qt5_NO_LINK_QTMAIN`` target property for
Qt 5 based targets or ``QT4_NO_LINK_QTMAIN`` target property for Qt 4 based
targets.

.. code-block:: cmake

  add_executable(myexe WIN32 main.cpp)
  target_link_libraries(myexe Qt4::QtGui)

  add_executable(myexe_no_qtmain WIN32 main_no_qtmain.cpp)
  set_property(TARGET main_no_qtmain PROPERTY QT4_NO_LINK_QTMAIN ON)
  target_link_libraries(main_no_qtmain Qt4::QtGui)
