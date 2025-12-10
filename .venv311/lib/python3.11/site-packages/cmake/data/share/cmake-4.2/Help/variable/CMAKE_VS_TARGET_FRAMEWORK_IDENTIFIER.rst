CMAKE_VS_TARGET_FRAMEWORK_IDENTIFIER
------------------------------------

.. versionadded:: 3.22

Visual Studio target framework identifier.

In some cases, the :ref:`Visual Studio Generators` may use an explicit value
for the MSBuild ``TargetFrameworkIdentifier`` setting in ``.csproj`` files.
CMake provides the chosen value in this variable.

See also :variable:`CMAKE_VS_TARGET_FRAMEWORK_VERSION` and
:variable:`CMAKE_VS_TARGET_FRAMEWORK_TARGETS_VERSION`.
