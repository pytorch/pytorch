VS_DOTNET_STARTUP_OBJECT
------------------------

.. versionadded:: 3.24

Sets the startup object property in Visual Studio .NET targets.
The property value defines a full qualified class name (including package
name), for example: ``MyCompany.Package.MyStarterClass``.

If the property is unset, Visual Studio uses the first matching
``static void Main(string[])`` function signature by default. When more
than one ``Main()`` method is available in the current project, the property
becomes mandatory for building the project.

This property only works for :ref:`Visual Studio Generators`;
it is ignored on other generators.

.. code-block:: cmake

  set_property(TARGET ${TARGET_NAME} PROPERTY
    VS_DOTNET_STARTUP_OBJECT "MyCompany.Package.MyStarterClass")
