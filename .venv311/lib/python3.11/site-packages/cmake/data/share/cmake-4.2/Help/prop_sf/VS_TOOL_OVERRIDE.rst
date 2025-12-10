VS_TOOL_OVERRIDE
----------------

.. versionadded:: 3.7

Override the MSBuild item type of a source file in a Visual Studio project.

Together with :prop_sf:`VS_SETTINGS`, this property can be used to configure
items for custom MSBuild tasks.

Setting the item type to ``None`` will exclude the file from the build.

.. versionchanged:: 3.31
  This property is honored for all source file types.
  Previously, it only worked for source types unknown to CMake.
