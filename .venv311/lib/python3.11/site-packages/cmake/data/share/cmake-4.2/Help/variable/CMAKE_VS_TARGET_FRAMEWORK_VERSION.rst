CMAKE_VS_TARGET_FRAMEWORK_VERSION
---------------------------------

.. versionadded:: 3.22

Visual Studio target framework version.

In some cases, the :ref:`Visual Studio Generators` may use an explicit value
for the MSBuild ``TargetFrameworkVersion`` setting in ``.csproj`` files.
CMake provides the chosen value in this variable.

See the :variable:`CMAKE_DOTNET_TARGET_FRAMEWORK_VERSION` variable
and :prop_tgt:`DOTNET_TARGET_FRAMEWORK_VERSION` target property to
specify custom ``TargetFrameworkVersion`` values for project targets.

See also :variable:`CMAKE_VS_TARGET_FRAMEWORK_IDENTIFIER` and
:variable:`CMAKE_VS_TARGET_FRAMEWORK_TARGETS_VERSION`.
