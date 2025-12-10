INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE
---------------------------------------

.. versionadded:: 3.24

List of libraries that consumers of this library should *not* treat
as direct link dependencies.

This target property may be set to *exclude* items from a dependent
target's final set of direct link dependencies.  This property is
processed after the :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT`
target property of all other dependencies of the dependent target, so
exclusion from direct link dependence takes priority over inclusion.

The initial set of a dependent target's direct link dependencies is
specified by its :prop_tgt:`LINK_LIBRARIES` target property.  Indirect
link dependencies are specified by the transitive closure of the direct
link dependencies' :prop_tgt:`INTERFACE_LINK_LIBRARIES` properties.
Any link dependency may specify additional direct link dependencies
using the :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` target property.
The set of direct link dependencies is then filtered to exclude items named
by any dependency's ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE`` target
property.

Excluding an item from a dependent target's direct link dependencies
does not mean the dependent target won't link the item.  The item
may still be linked as an indirect link dependency via the
:prop_tgt:`INTERFACE_LINK_LIBRARIES` property on other dependencies.

.. |INTERFACE_PROPERTY_LINK_DIRECT| replace:: ``INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE``
.. include:: include/INTERFACE_LINK_LIBRARIES_DIRECT.rst

See the :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` target property
documentation for more details and examples.
