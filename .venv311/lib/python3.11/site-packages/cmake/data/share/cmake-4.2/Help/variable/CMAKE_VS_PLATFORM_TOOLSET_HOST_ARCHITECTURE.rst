CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE
-------------------------------------------

.. versionadded:: 3.8

Visual Studio preferred tool architecture.

The :ref:`Visual Studio Generators` for VS 2013 and above support using
either the 32-bit or 64-bit host toolchains by specifying a ``host=x86``
or ``host=x64`` value in the :variable:`CMAKE_GENERATOR_TOOLSET` option.
CMake provides the selected toolchain architecture preference in this
variable (``x86``, ``x64``, ``ARM64`` or empty).
