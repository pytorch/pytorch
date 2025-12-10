LDFLAGS
-------

.. include:: include/ENV_VAR.rst

Will only be used by CMake on the first configuration to determine the default
linker flags, after which the value for ``LDFLAGS`` is stored in the cache
as :variable:`CMAKE_EXE_LINKER_FLAGS_INIT`,
:variable:`CMAKE_SHARED_LINKER_FLAGS_INIT`, and
:variable:`CMAKE_MODULE_LINKER_FLAGS_INIT`. For any configuration run
(including the first), the environment variable will be ignored if the
equivalent  ``CMAKE_<TYPE>_LINKER_FLAGS_INIT`` variable is defined.
