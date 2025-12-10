CMAKE_DOTNET_TARGET_FRAMEWORK_VERSION
-------------------------------------

.. versionadded:: 3.12

Default value for :prop_tgt:`DOTNET_TARGET_FRAMEWORK_VERSION`
property of targets.

This variable is used to initialize the
:prop_tgt:`DOTNET_TARGET_FRAMEWORK_VERSION` property on all
targets. See that target property for additional information. When set,
:variable:`CMAKE_DOTNET_TARGET_FRAMEWORK` takes precednece over this
variable. See that variable or the associated target property
:prop_tgt:`DOTNET_TARGET_FRAMEWORK` for additional information.


Setting ``CMAKE_DOTNET_TARGET_FRAMEWORK_VERSION`` may be necessary
when working with ``C#`` and newer .NET framework versions to
avoid referencing errors with the ``ALL_BUILD`` CMake target.

This variable is only evaluated for :ref:`Visual Studio Generators`
VS 2010 and above.
