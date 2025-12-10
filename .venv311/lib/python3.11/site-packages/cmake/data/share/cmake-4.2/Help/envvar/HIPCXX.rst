HIPCXX
------

.. versionadded:: 3.21

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``HIP`` language files. Will only be used by
CMake on the first configuration to determine ``HIP`` compiler, after which the
value for ``HIP`` is stored in the cache as
:variable:`CMAKE_HIP_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration
run (including the first), the environment variable will be ignored if the
:variable:`CMAKE_HIP_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.
