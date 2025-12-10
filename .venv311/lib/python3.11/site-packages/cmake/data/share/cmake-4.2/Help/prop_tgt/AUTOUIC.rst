AUTOUIC
-------

Should the target be processed with auto-uic (for Qt projects).

``AUTOUIC`` is a boolean specifying whether CMake will handle
the Qt ``uic`` code generator automatically, i.e. without having to use
commands like :module:`qt4_wrap_ui() <FindQt4>`, `qt5_wrap_ui()`_, etc.
Currently, Qt versions 4 to 6 are supported.

.. _`qt5_wrap_ui()`: https://doc.qt.io/archives/qt-5.15/qtwidgets-cmake-qt5-wrap-ui.html

This property is initialized by the value of the :variable:`CMAKE_AUTOUIC`
variable if it is set when a target is created.

When this property is ``ON``, CMake will scan the header and source files at
build time and invoke ``uic`` accordingly.


Header and source file processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At build time, CMake scans each header and source file from the
target's sources for include statements of the form

.. code-block:: c++

  #include "ui_<ui_base>.h"

Once such an include statement is found in a file, CMake searches for the
``uic`` input file ``<ui_base>.ui``

- in the vicinity of the file and
- in the :prop_tgt:`AUTOUIC_SEARCH_PATHS` of the target.

If the ``<ui_base>.ui`` file was found, ``uic`` is called on it to generate
``ui_<ui_base>.h`` in the directory

- ``<AUTOGEN_BUILD_DIR>/include`` for single configuration generators or in
- ``<AUTOGEN_BUILD_DIR>/include_<CONFIG>`` for
  :prop_gbl:`multi configuration <GENERATOR_IS_MULTI_CONFIG>` generators.

Where ``<AUTOGEN_BUILD_DIR>`` is the value of the target property
:prop_tgt:`AUTOGEN_BUILD_DIR`.

The include directory is automatically added to the target's
:prop_tgt:`INCLUDE_DIRECTORIES`.


Modifiers
^^^^^^^^^

:prop_tgt:`AUTOUIC_EXECUTABLE`:
The ``uic`` executable will be detected automatically, but can be forced to
a certain binary using this target property.

:prop_tgt:`AUTOUIC_OPTIONS`:
Additional command line options for ``uic`` can be set via this target
property.  The corresponding :prop_sf:`AUTOUIC_OPTIONS` source file property
can be used to specify options to be applied only to a specific
``<base_name>.ui`` file.

:prop_sf:`SKIP_AUTOUIC`:
Source files can be excluded from ``AUTOUIC`` processing by setting
this source file property.

:prop_sf:`SKIP_AUTOGEN`:
Source files can be excluded from :prop_tgt:`AUTOMOC`,
``AUTOUIC`` and :prop_tgt:`AUTORCC` processing by
setting this source file property.

:prop_gbl:`AUTOGEN_TARGETS_FOLDER`:
This global property can be used to group :prop_tgt:`AUTOMOC`,
``AUTOUIC`` and :prop_tgt:`AUTORCC` targets together in an IDE,
e.g.  in MSVS.

:variable:`CMAKE_GLOBAL_AUTOGEN_TARGET`:
A global ``autogen`` target, that depends on all :prop_tgt:`AUTOMOC` or
``AUTOUIC`` generated :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
targets in the project, will be generated when this variable is ``ON``.

:prop_tgt:`AUTOGEN_PARALLEL`:
This target property controls the number of ``moc`` or ``uic`` processes to
start in parallel during builds.

:prop_tgt:`AUTOGEN_COMMAND_LINE_LENGTH_MAX`:
This target property controls the limit when to use response files for
``moc`` or ``uic`` processes on Windows.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
