CMAKE_EXECUTABLE_ENABLE_EXPORTS
-------------------------------

.. versionadded:: 3.27

Specify whether executables export symbols for loadable modules.

This variable is used to initialize the :prop_tgt:`ENABLE_EXPORTS` target
property for executable targets when they are created by calls to the
:command:`add_executable` command.  See the property documentation for details.

This variable supersede the :variable:`CMAKE_ENABLE_EXPORTS` variable.
