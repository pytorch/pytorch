LINK_INTERFACE_LIBRARIES
------------------------

List public interface libraries for a shared library or executable.

By default linking to a shared library target transitively links to
targets with which the library itself was linked.  For an executable
with exports (see the :prop_tgt:`ENABLE_EXPORTS` target property) no
default transitive link dependencies are used.  This property replaces the default
transitive link dependencies with an explicit list.  When the target
is linked into another target using the :command:`target_link_libraries`
command, the libraries listed (and recursively
their link interface libraries) will be provided to the other target
also.  If the list is empty then no transitive link dependencies will
be incorporated when this target is linked into another target even if
the default set is non-empty.  This property is initialized by the
value of the :variable:`CMAKE_LINK_INTERFACE_LIBRARIES` variable if it is
set when a target is created.  This property is ignored for ``STATIC``
libraries.

This property is overridden by the :prop_tgt:`INTERFACE_LINK_LIBRARIES`
property if policy :policy:`CMP0022` is ``NEW``.

This property is deprecated.  Use :prop_tgt:`INTERFACE_LINK_LIBRARIES`
instead.

Creating Relocatable Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |INTERFACE_PROPERTY_LINK| replace:: ``LINK_INTERFACE_LIBRARIES``
.. include:: /include/INTERFACE_LINK_LIBRARIES_WARNING.rst
