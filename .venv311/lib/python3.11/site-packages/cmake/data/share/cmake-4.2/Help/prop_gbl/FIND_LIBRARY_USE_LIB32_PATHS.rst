FIND_LIBRARY_USE_LIB32_PATHS
----------------------------

.. versionadded:: 3.7

Whether the :command:`find_library` command should automatically search
``lib32`` directories.

``FIND_LIBRARY_USE_LIB32_PATHS`` is a boolean specifying whether the
:command:`find_library` command should automatically search the ``lib32``
variant of directories called ``lib`` in the search path when building 32-bit
binaries.

See also the :variable:`CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX` variable.
