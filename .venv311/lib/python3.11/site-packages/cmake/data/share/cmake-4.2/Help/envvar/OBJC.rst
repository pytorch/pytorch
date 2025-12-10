OBJC
----

.. versionadded:: 3.16.7

.. include:: include/ENV_VAR.rst

Preferred executable for compiling ``OBJC`` language files. Will only be used
by CMake on the first configuration to determine ``OBJC`` compiler, after
which the value for ``OBJC`` is stored in the cache as
:variable:`CMAKE_OBJC_COMPILER <CMAKE_<LANG>_COMPILER>`. For any configuration
run (including the first), the environment variable will be ignored if the
:variable:`CMAKE_OBJC_COMPILER <CMAKE_<LANG>_COMPILER>` variable is defined.

If ``OBJC`` is not defined, the :envvar:`CC` environment variable will
be checked instead.
