IMPORTED_LINK_INTERFACE_LIBRARIES
---------------------------------

Transitive link interface of an ``IMPORTED`` target.

Set this to the list of libraries whose interface is included when an
``IMPORTED`` library target is linked to another target.  The libraries
will be included on the link line for the target.  Unlike the
:prop_tgt:`LINK_INTERFACE_LIBRARIES` property, this property applies to all
imported target types, including ``STATIC`` libraries.  This property is
ignored for non-imported targets.

This property is ignored if the target also has a non-empty
:prop_tgt:`INTERFACE_LINK_LIBRARIES` property.

This property is deprecated.  Use :prop_tgt:`INTERFACE_LINK_LIBRARIES` instead.
