CMAKE_POLICY_VERSION_MINIMUM
----------------------------

.. versionadded:: 4.0

.. include:: include/ENV_VAR.rst

The default value for :variable:`CMAKE_POLICY_VERSION_MINIMUM` when there
is no explicit configuration given on the first run while creating a new
build tree.  On later runs in an existing build tree the value persists in
the cache as :variable:`CMAKE_POLICY_VERSION_MINIMUM`.
