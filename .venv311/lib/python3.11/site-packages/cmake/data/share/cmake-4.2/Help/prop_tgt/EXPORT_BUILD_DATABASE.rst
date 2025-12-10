EXPORT_BUILD_DATABASE
---------------------

.. versionadded:: 3.31

Enable/Disable output of a build database for a target.

This property is initialized by the value of the variable
:variable:`CMAKE_EXPORT_BUILD_DATABASE` if it is set when a target is created.

.. note::

   This property is meaningful only when experimental support for build
   databases has been enabled by the
   ``CMAKE_EXPERIMENTAL_EXPORT_BUILD_DATABASE`` gate.
