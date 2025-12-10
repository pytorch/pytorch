DOTNET_SDK
----------

.. versionadded:: 3.23

Specify the .NET SDK for C# projects.  For example: ``Microsoft.NET.Sdk``.

This property tells :ref:`Visual Studio Generators` for VS 2019 and
above to generate a .NET SDK-style project using the specified SDK.
The property is meaningful only to these generators, and only in C#
targets.  It is ignored for C++ projects, even if they are managed
(e.g. using :prop_tgt:`COMMON_LANGUAGE_RUNTIME`).

This property must be a non-empty string to generate .NET SDK-style projects.
CMake does not perform any validations for the value of the property.

This property may be initialized for all targets using the
:variable:`CMAKE_DOTNET_SDK` variable.

.. note::

  The :ref:`Visual Studio Generators` in this version of CMake have not
  yet learned to support :command:`add_custom_command` in .NET SDK-style
  projects.  It is currently an error to attach a custom command to a
  target with the ``DOTNET_SDK`` property set.
