CMAKE_CROSSCOMPILING_EMULATOR
-----------------------------

.. versionadded:: 3.28

.. include:: include/ENV_VAR.rst

The default value for :variable:`CMAKE_CROSSCOMPILING_EMULATOR` when there
is no explicit configuration given on the first run while creating a new
build tree.  On later runs in an existing build tree the value persists in
the cache as :variable:`CMAKE_CROSSCOMPILING_EMULATOR`.
