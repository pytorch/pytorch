EXPORT_NO_SYSTEM
----------------

.. versionadded:: 3.25

This property affects the behavior of the :command:`install(EXPORT)` and
:command:`export` commands when they install or export the target respectively.
When ``EXPORT_NO_SYSTEM`` is set to true, those commands generate an imported
target with :prop_tgt:`SYSTEM` property set to false.

See the :prop_tgt:`NO_SYSTEM_FROM_IMPORTED` target property to set this
behavior on the target *consuming* the include directories rather than the
one *providing* them.
