CMAKE_<LANG>_FLAGS_<CONFIG>_INIT
--------------------------------

.. versionadded:: 3.11

Value used to initialize the :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>` cache
entry the first time a build tree is configured for language ``<LANG>``.
This variable is meant to be set by a :variable:`toolchain file
<CMAKE_TOOLCHAIN_FILE>`.  CMake may prepend or append content to
the value based on the environment and target platform.

See also :variable:`CMAKE_<LANG>_FLAGS_INIT`.
