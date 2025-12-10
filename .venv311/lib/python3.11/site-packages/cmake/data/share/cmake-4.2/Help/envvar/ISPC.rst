ISPC
-------

.. versionadded:: 3.19

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``ISPC`` language files. Will only be used by
CMake on the first configuration to determine ``ISPC`` compiler, after which the
value for ``ISPC`` is stored in the cache as
:variable:`CMAKE_ISPC_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration
run (including the first), the environment variable will be ignored if the
:variable:`CMAKE_ISPC_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.
