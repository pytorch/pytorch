SKIP_AUTOMOC
------------

.. versionadded:: 3.8

Exclude the source file from :prop_tgt:`AUTOMOC` processing (for Qt projects).

For broader exclusion control see :prop_sf:`SKIP_AUTOGEN`.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(SOURCE file.h PROPERTY SKIP_AUTOMOC ON)
  # ...
