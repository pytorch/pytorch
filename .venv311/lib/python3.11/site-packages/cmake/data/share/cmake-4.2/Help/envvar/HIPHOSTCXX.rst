HIPHOSTCXX
-----------

.. versionadded:: 3.28

.. include:: include/ENV_VAR.rst

Preferred executable for compiling host code when compiling ``HIP``
language files with the NVIDIA CUDA Compiler. Will only be used by CMake
on the first configuration to determine ``HIP`` host compiler, after which
the value for ``HIPHOSTCXX`` is stored in the cache as
:variable:`CMAKE_HIP_HOST_COMPILER <CMAKE_<LANG>_HOST_COMPILER>`.

This environment variable is primarily meant for use with projects that
enable ``HIP`` as a first-class language.

.. note::

  Ignored when using :ref:`Visual Studio Generators`.
