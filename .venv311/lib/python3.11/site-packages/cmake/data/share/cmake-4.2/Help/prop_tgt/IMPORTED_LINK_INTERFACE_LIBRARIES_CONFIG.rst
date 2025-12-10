IMPORTED_LINK_INTERFACE_LIBRARIES_<CONFIG>
------------------------------------------

<CONFIG>-specific version of :prop_tgt:`IMPORTED_LINK_INTERFACE_LIBRARIES`.

Configuration names correspond to those provided by the project from
which the target is imported.  If set, this property completely
overrides the generic property for the named configuration.

This property is ignored if the target also has a non-empty
:prop_tgt:`INTERFACE_LINK_LIBRARIES` property.

This property is deprecated.  Use :prop_tgt:`INTERFACE_LINK_LIBRARIES` instead.
