CMAKE_AUTOMOC_INCLUDE_DIRECTORIES
---------------------------------

.. versionadded:: 4.1

Specifies zero or more include directories for AUTOMOC to pass explicitly to
the Qt Meta‑Object Compiler (``moc``) instead of automatically discovering
each target's include directories.

The directories listed here will replace any include paths discovered from
target properties such as :prop_tgt:`INCLUDE_DIRECTORIES`.

This variable is used to initialize the :prop_tgt:`AUTOMOC_INCLUDE_DIRECTORIES`
property on all the targets.  See that target property for additional
information.
