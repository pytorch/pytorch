CMAKE_APPLE_SILICON_PROCESSOR
-----------------------------

.. versionadded:: 3.19.2

On Apple Silicon hosts running macOS, set this variable to tell
CMake what architecture to use for :variable:`CMAKE_HOST_SYSTEM_PROCESSOR`.
The value must be either ``arm64`` or ``x86_64``.

The value of this variable should never be modified by project code.
It is meant to be set as a cache entry provided by the user,
e.g. via ``-DCMAKE_APPLE_SILICON_PROCESSOR=...``.

See also the :envvar:`CMAKE_APPLE_SILICON_PROCESSOR` environment variable.
