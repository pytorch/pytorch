CMAKE_AUTORCC_OPTIONS
---------------------

Additional options for ``rcc`` when using :variable:`CMAKE_AUTORCC`.

This variable is used to initialize the :prop_tgt:`AUTORCC_OPTIONS` property on
all the targets.  See that target property for additional information.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set(CMAKE_AUTORCC_OPTIONS "--compress;9")
  # ...
