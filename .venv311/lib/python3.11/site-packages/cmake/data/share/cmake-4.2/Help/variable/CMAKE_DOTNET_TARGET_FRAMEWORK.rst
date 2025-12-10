CMAKE_DOTNET_TARGET_FRAMEWORK
-----------------------------

.. versionadded:: 3.17

Default value for :prop_tgt:`DOTNET_TARGET_FRAMEWORK` property of
targets.

This variable is used to initialize the
:prop_tgt:`DOTNET_TARGET_FRAMEWORK` property on all targets. See that
target property for additional information.

Setting ``CMAKE_DOTNET_TARGET_FRAMEWORK`` may be necessary
when working with ``C#`` and newer .NET framework versions to
avoid referencing errors with the ``ALL_BUILD`` CMake target.

This variable is only evaluated for :ref:`Visual Studio Generators`
VS 2010 and above.
