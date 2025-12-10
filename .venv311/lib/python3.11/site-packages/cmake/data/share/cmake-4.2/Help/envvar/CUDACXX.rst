CUDACXX
-------

.. versionadded:: 3.8

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``CUDA`` language files. Will only be used by
CMake on the first configuration to determine ``CUDA`` compiler, after which the
value for ``CUDA`` is stored in the cache as
:variable:`CMAKE_CUDA_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration
run (including the first), the environment variable will be ignored if the
:variable:`CMAKE_CUDA_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.

.. note::
  Options that are required to make the compiler work correctly can be included;
  they can not be changed.

.. code-block:: console

  $ export CUDACXX="custom-compiler --arg1 --arg2"
