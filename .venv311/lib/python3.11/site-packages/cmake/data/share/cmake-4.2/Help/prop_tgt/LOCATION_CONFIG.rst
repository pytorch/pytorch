LOCATION_<CONFIG>
-----------------

Read-only property providing a target location on disk.

A read-only property that indicates where a target's main file is
located on disk for the configuration ``<CONFIG>``.  The property is
defined only for library and executable targets.  An imported target
may provide a set of configurations different from that of the
importing project.  By default CMake looks for an exact-match but
otherwise uses an arbitrary available configuration.  Use the
:prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` property to map imported
configurations explicitly.

Do not set properties that affect the location of a target after
reading this property.  These include properties whose names match
``(RUNTIME|LIBRARY|ARCHIVE)_OUTPUT_(NAME|DIRECTORY)(_<CONFIG>)?``,
``(IMPLIB_)?(PREFIX|SUFFIX)``, or  :prop_tgt:`LINKER_LANGUAGE`.
Failure to follow this rule is not diagnosed and leaves
the location of the target undefined.
