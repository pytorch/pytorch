CMAKE_AUTOGEN_VERBOSE
---------------------

.. versionadded:: 3.13

Sets the verbosity of :prop_tgt:`AUTOMOC`, :prop_tgt:`AUTOUIC` and
:prop_tgt:`AUTORCC`.  A positive integer value or a true boolean value
lets the ``AUTO*`` generators output additional processing information.

Setting ``CMAKE_AUTOGEN_VERBOSE`` has the same effect
as setting the ``VERBOSE`` environment variable during
generation (e.g. by calling ``make VERBOSE=1``).
The extra verbosity is limited to the ``AUTO*`` generators though.

By default ``CMAKE_AUTOGEN_VERBOSE`` is unset.
