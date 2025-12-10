INTERFACE_LINK_DEPENDS
----------------------

.. versionadded:: 3.13

Additional public interface files on which a target binary depends for linking.

This property is supported only by :generator:`Ninja` and
:ref:`Makefile Generators`.
It is intended to specify dependencies on "linker scripts" for
custom Makefile link rules.

When target dependencies are specified using :command:`target_link_libraries`,
CMake will read this property from all target dependencies to determine the
build properties of the consumer.

Contents of ``INTERFACE_LINK_DEPENDS`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
-manual for more on defining buildsystem properties.

Link dependency files usage requirements commonly differ between the build-tree
and the install-tree.  The ``BUILD_INTERFACE`` and ``INSTALL_INTERFACE``
generator expressions can be used to describe separate usage requirements
based on the usage location.  Relative paths are allowed within the
``INSTALL_INTERFACE`` expression and are interpreted relative to the
installation prefix.  For example:

.. code-block:: cmake

  set_property(TARGET mylib PROPERTY INTERFACE_LINK_DEPENDS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mylinkscript>
    $<INSTALL_INTERFACE:mylinkscript>  # <prefix>/mylinkscript
  )
