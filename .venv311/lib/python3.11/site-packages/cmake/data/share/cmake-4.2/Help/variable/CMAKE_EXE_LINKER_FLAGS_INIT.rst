CMAKE_EXE_LINKER_FLAGS_INIT
---------------------------

.. versionadded:: 3.7

Value used to initialize the :variable:`CMAKE_EXE_LINKER_FLAGS`
cache entry the first time a build tree is configured.
This variable is meant to be set by a :variable:`toolchain file
<CMAKE_TOOLCHAIN_FILE>`.  CMake may prepend or append content to
the value based on the environment and target platform.

See also the configuration-specific variable
:variable:`CMAKE_EXE_LINKER_FLAGS_<CONFIG>_INIT`.
