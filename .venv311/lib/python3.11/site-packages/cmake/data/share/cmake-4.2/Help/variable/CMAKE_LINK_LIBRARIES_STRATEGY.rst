CMAKE_LINK_LIBRARIES_STRATEGY
-----------------------------

.. versionadded:: 3.31

Specify a strategy for ordering targets' direct link dependencies
on linker command lines.

If set, this variable acts as the default value for the
:prop_tgt:`LINK_LIBRARIES_STRATEGY` target property when a target is created.
Set that property directly to specify a strategy for a single target.
