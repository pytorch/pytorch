CMAKE_<LANG>_LINKER_LAUNCHER
----------------------------

.. versionadded:: 3.21

.. include:: include/ENV_VAR.rst

Default launcher to use when linking a target of the specified language. Will
only be used by CMake to initialize the variable on the first configuration.
Afterwards, it is available through the cache setting of the variable of the
same name. For any configuration run (including the first), the environment
variable will be ignored if the :variable:`CMAKE_<LANG>_LINKER_LAUNCHER`
variable is defined.
