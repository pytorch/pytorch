CMAKE_FRAMEWORK_PATH
--------------------

.. include:: include/ENV_VAR.rst

The ``CMAKE_FRAMEWORK_PATH`` environment variable may be set to a list of
directories to be searched for macOS frameworks by the :command:`find_library`,
:command:`find_package`, :command:`find_path` and :command:`find_file` commands.


This variable may hold a single directory or a list of directories separated
by ``:`` on UNIX or ``;`` on Windows (the same as the ``PATH`` environment
variable convention on those platforms).

See also the :variable:`CMAKE_FRAMEWORK_PATH` CMake variable.
