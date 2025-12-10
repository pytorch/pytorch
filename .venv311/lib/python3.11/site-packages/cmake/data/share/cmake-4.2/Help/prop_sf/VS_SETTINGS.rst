VS_SETTINGS
-----------

.. versionadded:: 3.18

Add arbitrary MSBuild item metadata to a file.

This property accepts a list of ``Key=Value`` pairs. The Visual Studio
generator will add these key-value pairs as item metadata to the file.
:manual:`Generator expressions <cmake-generator-expressions(7)>` are supported.

For example:

.. code-block:: cmake

  set_property(SOURCE file.hlsl PROPERTY VS_SETTINGS "Key=Value" "Key2=Value2")

will set the ``file.hlsl`` item metadata as follows:

.. code-block:: xml

  <FXCompile Include="source_path\file.hlsl">
    <Key>Value</Key>
    <Key2>Value2</Key2>
  </FXCompile>

Together with :prop_sf:`VS_TOOL_OVERRIDE`, this property can be used to
configure items for custom MSBuild tasks.

Adding the metadata ``ExcludedFromBuild=true`` will exclude the file from
the build.

.. versionchanged:: 3.22
  This property is honored for all source file types.
  Previously, it only worked for source types unknown to CMake.
