VS_PROJECT_IMPORT
-----------------

.. versionadded:: 3.15

Visual Studio managed project imports

Adds to a generated Visual Studio project one or more paths to ``.props``
files needed when building projects from some NuGet packages.

For example:

.. code-block:: cmake

  set_property(TARGET myTarget PROPERTY VS_PROJECT_IMPORT
    "my_packages_path/PackageA.1.0.0/build/PackageA.props"
    "my_packages_path/PackageB.1.0.0/build/PackageB.props"
  )
