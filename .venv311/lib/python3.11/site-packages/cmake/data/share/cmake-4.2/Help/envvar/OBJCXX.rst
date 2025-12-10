OBJCXX
------

.. versionadded:: 3.16.7

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``OBJCXX`` language files. Will only be used
by CMake on the first configuration to determine ``OBJCXX`` compiler, after
which the value for ``OBJCXX`` is stored in the cache as
:variable:`CMAKE_OBJCXX_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration
run (including the first), the environment variable will be ignored if the
:variable:`CMAKE_OBJCXX_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.

If ``OBJCXX`` is not defined, the :envvar:`CXX` environment variable will
be checked instead.
