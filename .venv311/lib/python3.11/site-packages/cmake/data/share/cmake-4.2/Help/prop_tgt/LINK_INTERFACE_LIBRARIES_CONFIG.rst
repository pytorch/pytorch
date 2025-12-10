LINK_INTERFACE_LIBRARIES_<CONFIG>
---------------------------------

Per-configuration list of public interface libraries for a target.

This is the configuration-specific version of
:prop_tgt:`LINK_INTERFACE_LIBRARIES`.  If set, this property completely
overrides the generic property for the named configuration.

This property is overridden by the :prop_tgt:`INTERFACE_LINK_LIBRARIES`
property if policy :policy:`CMP0022` is ``NEW``.

This property is deprecated.  Use :prop_tgt:`INTERFACE_LINK_LIBRARIES`
instead.

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: ``LINK_INTERFACE_LIBRARIES_<CONFIG>``
.. include:: /include/INTERFACE_LINK_LIBRARIES_WARNING.rst
