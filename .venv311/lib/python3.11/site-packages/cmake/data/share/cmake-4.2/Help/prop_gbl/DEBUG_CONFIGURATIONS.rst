DEBUG_CONFIGURATIONS
--------------------

This property specifies which :ref:`Build Configurations` are for debugging.

The value must be a :ref:`semicolon-separated list <CMake Language Lists>` of
build configuration names.
Currently this property is used only by the :command:`target_link_libraries`
command.  Additional uses may be defined in the future.

This property must be set at the top level of the project and before
the first :command:`target_link_libraries` command invocation.  If any entry in
the list does not match a valid configuration for the project, the
behavior is undefined.

By default, this property is **not set**.

Examples
^^^^^^^^

The following example adds a custom configuration to non-optimized debug
configurations while preserving any existing ones.  If the project uses the
default ``Debug`` configuration, it should be included as well.

.. code-block:: cmake

  set_property(GLOBAL APPEND PROPERTY DEBUG_CONFIGURATIONS Debug CustomBuild)

See Also
^^^^^^^^

* The :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` target property, which maps build
  configurations when linking to :ref:`Imported Targets` that have the
  :prop_tgt:`IMPORTED_CONFIGURATIONS` property set.
