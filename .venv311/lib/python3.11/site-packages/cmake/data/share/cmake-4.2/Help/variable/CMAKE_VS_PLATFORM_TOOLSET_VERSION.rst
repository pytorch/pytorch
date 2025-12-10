CMAKE_VS_PLATFORM_TOOLSET_VERSION
---------------------------------

.. versionadded:: 3.12

Visual Studio Platform Toolset version.

The :ref:`Visual Studio Generators` for VS 2017 and above allow to
select minor versions of the same toolset. The toolset version number
may be specified by a field in :variable:`CMAKE_GENERATOR_TOOLSET` of
the form ``version=14.11``. If none is specified CMake will choose a default
toolset. The value may be empty if no minor version was selected and the
default is used.

If the value is not empty, it is the version number that MSBuild uses in
its ``Microsoft.VCToolsVersion.*.props`` file names.

.. versionadded:: 3.19.7

   VS 16.9's toolset may also be specified as ``14.28.16.9`` because
   VS 16.10 uses the file name ``Microsoft.VCToolsVersion.14.28.16.9.props``.

Three-Component MSVC Toolset Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.19.7

The ``version=`` field may be given a three-component toolset version
such as ``14.28.29910``, and CMake will convert it to the name used by
MSBuild ``Microsoft.VCToolsVersion.*.props`` files.  This is useful
to distinguish between VS 16.8's ``14.28.29333`` toolset and VS 16.9's
``14.28.29910`` toolset.  It also matches ``vcvarsall``'s ``-vcvars_ver=``
behavior.
