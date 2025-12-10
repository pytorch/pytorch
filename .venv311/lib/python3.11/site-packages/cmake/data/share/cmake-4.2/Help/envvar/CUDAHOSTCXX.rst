CUDAHOSTCXX
-----------

.. versionadded:: 3.8

.. include:: include/ENV_VAR.rst

Preferred executable for compiling host code when compiling ``CUDA``
language files. Will only be used by CMake on the first configuration to
determine ``CUDA`` host compiler, after which the value for ``CUDAHOSTCXX`` is
stored in the cache as
:variable:`CMAKE_CUDA_HOST_COMPILER <CMAKE_<LANG>_HOST_COMPILER>`.
This environment variable is preferred over
:variable:`CMAKE_CUDA_HOST_COMPILER <CMAKE_<LANG>_HOST_COMPILER>`.

This environment variable is primarily meant for use with projects that
enable ``CUDA`` as a first-class language.

.. note::

  Ignored when using :ref:`Visual Studio Generators`.

.. versionadded:: 3.13
  The :module:`FindCUDA`
  module will use this variable to initialize its ``CUDA_HOST_COMPILER`` setting.
