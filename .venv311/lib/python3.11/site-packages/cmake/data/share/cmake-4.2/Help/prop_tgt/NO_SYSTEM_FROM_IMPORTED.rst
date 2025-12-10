NO_SYSTEM_FROM_IMPORTED
-----------------------

Do not treat include directories from the interfaces of consumed
:ref:`imported targets` as system directories.

When the consumed target's :prop_tgt:`SYSTEM` property is set to true, the
contents of the :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` target property are
treated as system includes or, on Apple platforms, when the target is a
framework, it will be treated as system.  By default, :prop_tgt:`SYSTEM` is
true for imported targets and false for other target types.  If the
``NO_SYSTEM_FROM_IMPORTED`` property is set to true on a *consuming* target,
compilation of sources in that consuming target will not treat the contents of
the :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` of consumed imported targets as
system includes, even if that imported target's :prop_tgt:`SYSTEM` property
is false.

Directories listed in the :prop_tgt:`INTERFACE_SYSTEM_INCLUDE_DIRECTORIES`
property of consumed targets are not affected by ``NO_SYSTEM_FROM_IMPORTED``.
Those directories will always be treated as system include directories by
consumers.

This property is initialized by the value of the
:variable:`CMAKE_NO_SYSTEM_FROM_IMPORTED` variable if it is set when a target
is created.

See the :prop_tgt:`EXPORT_NO_SYSTEM` target property to set this behavior
on the target providing the include directories rather than the target
consuming them.
