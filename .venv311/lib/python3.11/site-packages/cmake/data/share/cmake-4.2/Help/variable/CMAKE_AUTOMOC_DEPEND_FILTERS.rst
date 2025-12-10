CMAKE_AUTOMOC_DEPEND_FILTERS
----------------------------

.. versionadded:: 3.9

Filter definitions used by :variable:`CMAKE_AUTOMOC`
to extract file names from source code as additional dependencies
for the ``moc`` file.

This variable is used to initialize the :prop_tgt:`AUTOMOC_DEPEND_FILTERS`
property on all the targets. See that target property for additional
information.

By default it is empty.
