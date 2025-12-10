CMAKE_LINK_LIBRARIES_ONLY_TARGETS
---------------------------------

.. versionadded:: 3.23

Set this variable to initialize the :prop_tgt:`LINK_LIBRARIES_ONLY_TARGETS`
property of non-imported targets when they are created.  Setting it to true
enables an additional check that all items named by
:command:`target_link_libraries` that can be target names are actually names
of existing targets.  See the target property documentation for details.
