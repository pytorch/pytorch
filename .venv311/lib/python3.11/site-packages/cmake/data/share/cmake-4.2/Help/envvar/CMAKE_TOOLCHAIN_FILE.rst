CMAKE_TOOLCHAIN_FILE
--------------------

.. versionadded:: 3.21

.. include:: include/ENV_VAR.rst

The ``CMAKE_TOOLCHAIN_FILE`` environment variable specifies a default value
for the :variable:`CMAKE_TOOLCHAIN_FILE` variable when there is no explicit
configuration given on the first run while creating a new build tree.
On later runs in an existing build tree the value persists in the cache
as :variable:`CMAKE_TOOLCHAIN_FILE`.
