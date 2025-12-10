LINK_LIBRARIES
--------------

List of direct link dependencies.

This property specifies the list of libraries or targets which will be
used for linking.  In addition to accepting values from the
:command:`target_link_libraries` command, values may be set directly on
any target using the :command:`set_property` command.

The value of this property is used by the generators to construct the
link rule for the target.  The direct link dependencies are linked first,
followed by indirect dependencies from the transitive closure of the
direct dependencies' :prop_tgt:`INTERFACE_LINK_LIBRARIES` properties.
See policy :policy:`CMP0022`.

Contents of ``LINK_LIBRARIES`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>` with the
syntax ``$<...>``.  Policy :policy:`CMP0131` affects the behavior of the
:genex:`LINK_ONLY` generator expression for this property.

See the :manual:`cmake-buildsystem(7)` manual for more on defining
buildsystem properties.

.. include:: include/LINK_LIBRARIES_INDIRECTION.rst

In advanced use cases, the list of direct link dependencies specified
by this property may be updated by usage requirements from dependencies.
See the :prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` and
:prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE` target properties.

See the :variable:`CMAKE_LINK_LIBRARIES_STRATEGY` variable and
corresponding :prop_tgt:`LINK_LIBRARIES_STRATEGY` target property
for details on how CMake orders direct link dependencies on linker
command lines.

.. include:: ../command/include/LINK_LIBRARIES_LINKER.rst
