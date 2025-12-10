CMAKE_EXPORT_BUILD_DATABASE
---------------------------

.. versionadded:: 3.31

.. include:: include/ENV_VAR.rst

The default value for :variable:`CMAKE_EXPORT_BUILD_DATABASE` when there is no
explicit configuration given on the first run while creating a new build tree.
On later runs in an existing build tree the value persists in the cache as
:variable:`CMAKE_EXPORT_BUILD_DATABASE`.

.. note::

   This variable is meaningful only when experimental support for build
   databases has been enabled by the
   ``CMAKE_EXPERIMENTAL_EXPORT_BUILD_DATABASE`` gate.
