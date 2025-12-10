CMAKE_EXPORT_COMPILE_COMMANDS
-----------------------------

.. versionadded:: 3.17

.. include:: include/ENV_VAR.rst

The default value for :variable:`CMAKE_EXPORT_COMPILE_COMMANDS` when there
is no explicit configuration given on the first run while creating a new
build tree.  On later runs in an existing build tree the value persists in
the cache as :variable:`CMAKE_EXPORT_COMPILE_COMMANDS`.
