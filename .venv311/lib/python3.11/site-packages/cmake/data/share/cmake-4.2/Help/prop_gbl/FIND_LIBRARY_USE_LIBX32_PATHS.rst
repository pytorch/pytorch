FIND_LIBRARY_USE_LIBX32_PATHS
-----------------------------

.. versionadded:: 3.9

Whether the :command:`find_library` command should automatically search
``libx32`` directories.

``FIND_LIBRARY_USE_LIBX32_PATHS`` is a boolean specifying whether the
:command:`find_library` command should automatically search the ``libx32``
variant of directories called ``lib`` in the search path when building
x32-abi binaries.

See also the :variable:`CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX` variable.
