CMAKE_AUTOUIC_OPTIONS
---------------------

Additional options for ``uic`` when using :variable:`CMAKE_AUTOUIC`.

This variable is used to initialize the :prop_tgt:`AUTOUIC_OPTIONS` property on
all the targets.  See that target property for additional information.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(CMAKE_AUTOUIC_OPTIONS "--no-protection")
  # ...
