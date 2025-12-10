CMAKE_<LANG>_COMPILER_LAUNCHER
------------------------------

.. versionadded:: 3.17

.. include:: include/ENV_VAR.rst

Default compiler launcher to use for the specified language. Will only be used
by CMake to initialize the variable on the first configuration. Afterwards, it
is available through the cache setting of the variable of the same name. For
any configuration run (including the first), the environment variable will be
ignored if the :variable:`CMAKE_<LANG>_COMPILER_LAUNCHER` variable is defined.
