CMAKE_APPLE_SILICON_PROCESSOR
-----------------------------

.. versionadded:: 3.19.2

.. include:: include/ENV_VAR.rst

On Apple Silicon hosts running macOS, set this environment variable to tell
CMake what architecture to use for :variable:`CMAKE_HOST_SYSTEM_PROCESSOR`.
The value must be either ``arm64`` or ``x86_64``.

The :variable:`CMAKE_APPLE_SILICON_PROCESSOR` normal variable, if set,
overrides this environment variable.
