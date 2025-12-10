CMAKE_APPBUNDLE_PATH
--------------------

.. include:: include/ENV_VAR.rst

The ``CMAKE_APPBUNDLE_PATH`` environment variable may be set to a list of
directories to be searched for macOS application bundles
by the :command:`find_program` and :command:`find_package` commands.

This variable may hold a single directory or a list of directories separated
by ``:`` on UNIX or ``;`` on Windows (the same as the ``PATH`` environment
variable convention on those platforms).

See also the :variable:`CMAKE_APPBUNDLE_PATH` CMake variable.
