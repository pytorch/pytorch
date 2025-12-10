SYSTEM
------

.. versionadded:: 3.25

Specifies that a target is a system target.  This has the following
effects:

* Entries of :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` are treated as
  system include directories when compiling consumers.
  Entries of :prop_tgt:`INTERFACE_SYSTEM_INCLUDE_DIRECTORIES` are not
  affected, and will always be treated as system include directories.
* On Apple platforms, If the :prop_tgt:`FRAMEWORK` target property is true,
  the frameworks directory is treated as system.

For imported targets, this property defaults to true, which means
that their :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` and, if the
:prop_tgt:`FRAMEWORK` target property is true, frameworks directory are
treated as system directories by default.  If their ``SYSTEM`` property is
false, then their :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` as well as
frameworks will not be treated as system.  Use the :prop_tgt:`EXPORT_NO_SYSTEM`
property to change how a target's ``SYSTEM`` property is set when it is
installed.

For non-imported targets, this target property is initialized from
the :prop_dir:`SYSTEM` directory property when the target is created.
