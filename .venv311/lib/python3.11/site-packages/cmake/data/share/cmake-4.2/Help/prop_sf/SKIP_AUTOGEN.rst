SKIP_AUTOGEN
------------

.. versionadded:: 3.8

Exclude the source file from :prop_tgt:`AUTOMOC`, :prop_tgt:`AUTOUIC` and
:prop_tgt:`AUTORCC` processing (for Qt projects).

For finer exclusion control see :prop_sf:`SKIP_AUTOMOC`,
:prop_sf:`SKIP_AUTOUIC` and :prop_sf:`SKIP_AUTORCC`.

EXAMPLE
^^^^^^^

.. code-block:: cmake

  # ...
  set_property(SOURCE file.h PROPERTY SKIP_AUTOGEN ON)
  # ...
