INTERFACE_LINK_LIBRARIES
------------------------

List public interface libraries for a library.

This property contains the list of transitive link dependencies.  When
the target is linked into another target using the
:command:`target_link_libraries` command, the libraries listed (and
recursively their link interface libraries) will be provided to the
other target also.  This property is overridden by the
:prop_tgt:`LINK_INTERFACE_LIBRARIES` or
:prop_tgt:`LINK_INTERFACE_LIBRARIES_<CONFIG>` property if policy
:policy:`CMP0022` is ``OLD`` or unset.

The value of this property is used by the generators when constructing
the link rule for a dependent target.  A dependent target's direct
link dependencies, specified by its :prop_tgt:`LINK_LIBRARIES` target
property, are linked first, followed by indirect dependencies from the
transitive closure of the direct dependencies'
``INTERFACE_LINK_LIBRARIES`` properties.  See policy :policy:`CMP0022`.

Contents of ``INTERFACE_LINK_LIBRARIES`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.

.. include:: include/LINK_LIBRARIES_INDIRECTION.rst

``INTERFACE_LINK_LIBRARIES`` adds transitive link dependencies for a
target's dependents.  In advanced use cases, one may update the
direct link dependencies of a target's dependents by using the
:prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` and
:prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE` target properties.

.. include:: ../command/include/LINK_LIBRARIES_LINKER.rst

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: ``INTERFACE_LINK_LIBRARIES``
.. include:: /include/INTERFACE_LINK_LIBRARIES_WARNING.rst
