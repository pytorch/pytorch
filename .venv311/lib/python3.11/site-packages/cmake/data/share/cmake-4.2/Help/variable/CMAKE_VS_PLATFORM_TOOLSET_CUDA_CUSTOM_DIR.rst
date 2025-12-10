CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR
-----------------------------------------

.. versionadded:: 3.16

Path to standalone NVIDIA CUDA Toolkit (eg. extracted from installer).

The :ref:`Visual Studio Generators` for VS 2010 and above support using
a standalone (non-installed) NVIDIA CUDA toolkit.  The path
may be specified by a field in :variable:`CMAKE_GENERATOR_TOOLSET` of
the form ``cuda=C:\path\to\cuda``.  The given directory must at least
contain the nvcc compiler in path ``.\bin`` and must provide Visual Studio
integration files in path ``.\extras\visual_studio_integration\
MSBuildExtensions\``. One can create a standalone CUDA toolkit directory by
either opening a installer with 7zip or copying the files that are extracted
by the running installer. The value may be empty if no path to a standalone
CUDA Toolkit was specified.
