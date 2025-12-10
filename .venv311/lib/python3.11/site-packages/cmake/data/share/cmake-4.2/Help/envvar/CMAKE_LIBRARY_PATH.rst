CMAKE_LIBRARY_PATH
------------------

.. include:: include/ENV_VAR.rst

The ``CMAKE_LIBRARY_PATH`` environment variable may be set to a list of
directories to be searched by the :command:`find_library` command.

This variable may hold a single directory or a list of directories separated
by ``:`` on UNIX or ``;`` on Windows (the same as the ``PATH`` environment
variable convention on those platforms).

See also the :variable:`CMAKE_LIBRARY_PATH` CMake variable.
