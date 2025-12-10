AUTOUIC_OPTIONS
---------------

Additional options for ``uic`` when using :prop_tgt:`AUTOUIC`

This property holds additional command line options
which will be used when ``uic`` is executed during the build via
:prop_tgt:`AUTOUIC`, i.e. it is equivalent to the optional ``OPTIONS``
argument of the :module:`qt4_wrap_ui() <FindQt4>` macro.

By default it is empty.

The options set on the ``.ui`` source file may override
:prop_tgt:`AUTOUIC_OPTIONS` set on the target.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(SOURCE widget.ui PROPERTY AUTOUIC_OPTIONS "--no-protection")
  # ...
