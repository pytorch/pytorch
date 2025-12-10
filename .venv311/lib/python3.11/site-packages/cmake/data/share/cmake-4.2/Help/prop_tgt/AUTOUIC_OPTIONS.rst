AUTOUIC_OPTIONS
---------------

.. versionadded:: 3.0

Additional options for ``uic`` when using :prop_tgt:`AUTOUIC`

This property holds additional command line options which will be used when
``uic`` is executed during the build via :prop_tgt:`AUTOUIC`, i.e. it is
equivalent to the optional ``OPTIONS`` argument of the
:module:`qt4_wrap_ui() <FindQt4>` macro.

This property is initialized by the value of the
:variable:`CMAKE_AUTOUIC_OPTIONS` variable if it is set when a target is
created, or an empty string otherwise.

The options set on the target may be overridden by :prop_sf:`AUTOUIC_OPTIONS`
set on the ``.ui`` source file.

This property may use "generator expressions" with the syntax ``$<...>``.
See the :manual:`cmake-generator-expressions(7)` manual for available
expressions.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(TARGET tgt PROPERTY AUTOUIC_OPTIONS "--no-protection")
  # ...
