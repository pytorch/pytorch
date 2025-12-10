AUTORCC_OPTIONS
---------------

Additional options for ``rcc`` when using :prop_tgt:`AUTORCC`

This property holds additional command line options which will be used
when ``rcc`` is executed during the build via :prop_tgt:`AUTORCC`,
i.e. it is equivalent to the optional ``OPTIONS`` argument of the
:module:`qt4_add_resources() <FindQt4>` macro.

This property is initialized by the value of the
:variable:`CMAKE_AUTORCC_OPTIONS` variable if it is set when a target is
created, or an empty string otherwise.

The options set on the target may be overridden by :prop_sf:`AUTORCC_OPTIONS`
set on the ``.qrc`` source file.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(TARGET tgt PROPERTY AUTORCC_OPTIONS "--compress;9")
  # ...
