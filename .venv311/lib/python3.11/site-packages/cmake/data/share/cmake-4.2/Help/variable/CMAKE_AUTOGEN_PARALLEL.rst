CMAKE_AUTOGEN_PARALLEL
----------------------

.. versionadded:: 3.11

Number of parallel ``moc`` or ``uic`` processes to start when using
:prop_tgt:`AUTOMOC` and :prop_tgt:`AUTOUIC`.

This variable is used to initialize the :prop_tgt:`AUTOGEN_PARALLEL` property
on all the targets.  See that target property for additional information.

By default ``CMAKE_AUTOGEN_PARALLEL`` is unset.
