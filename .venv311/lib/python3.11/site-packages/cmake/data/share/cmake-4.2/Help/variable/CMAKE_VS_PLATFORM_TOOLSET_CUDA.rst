CMAKE_VS_PLATFORM_TOOLSET_CUDA
------------------------------

.. versionadded:: 3.9

NVIDIA CUDA Toolkit version whose Visual Studio toolset to use.

The :ref:`Visual Studio Generators` for VS 2010 and above support using
a CUDA toolset provided by a CUDA Toolkit.  The toolset version number
may be specified by a field in :variable:`CMAKE_GENERATOR_TOOLSET` of
the form ``cuda=8.0``. Or it is automatically detected if a path to
a standalone CUDA directory is specified in the form ``cuda=C:\path\to\cuda``.
If none is specified CMake will choose a default version.
CMake provides the selected CUDA toolset version in this variable.
The value may be empty if no CUDA Toolkit with Visual Studio integration
is installed.
