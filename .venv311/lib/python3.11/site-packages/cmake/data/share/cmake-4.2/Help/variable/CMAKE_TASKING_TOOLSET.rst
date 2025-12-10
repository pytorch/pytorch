CMAKE_TASKING_TOOLSET
---------------------

.. versionadded:: 3.25

Select the Tasking toolset which provides the compiler

Architecture compilers are provided by different toolchains with
incompatible versioning schemes.  Set this variable in a
:variable:`toolchain file <CMAKE_TOOLCHAIN_FILE>` so CMake can detect
the compiler features correctly. If no toolset is specified,
``Standalone`` is assumed.

Due to the different versioning schemes, the compiler version
(:variable:`CMAKE_<LANG>_COMPILER_VERSION`) depends on the toolset and
architecture in use. If projects can be built with multiple toolsets or
architectures, the specified ``CMAKE_TASKING_TOOLSET`` and the
automatically determined :variable:`CMAKE_<LANG>_COMPILER_ARCHITECTURE_ID`
must be taken into account when comparing against the
:variable:`CMAKE_<LANG>_COMPILER_VERSION`.

``TriCore``
  Compilers are provided by the TriCore toolset.

``SmartCode``
  Compilers are provided by the SmartCode toolset.

``Standalone``
  Compilers are provided by the standalone toolsets.

  .. note::

    For the TriCore architecture, the compiler from the TriCore toolset is
    selected as standalone compiler.
