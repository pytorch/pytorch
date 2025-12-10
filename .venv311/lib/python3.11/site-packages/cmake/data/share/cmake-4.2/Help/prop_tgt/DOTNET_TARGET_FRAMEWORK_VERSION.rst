DOTNET_TARGET_FRAMEWORK_VERSION
-------------------------------

.. versionadded:: 3.12

Specify the .NET target framework version.

Used to specify the .NET target framework version for C++/CLI and C#.
For example: ``v4.5``.

This property is only evaluated for :ref:`Visual Studio Generators`
VS 2010 and above.

To initialize this variable for all targets set
:variable:`CMAKE_DOTNET_TARGET_FRAMEWORK` or
:variable:`CMAKE_DOTNET_TARGET_FRAMEWORK_VERSION`. If both are set,
the latter is ignored.
