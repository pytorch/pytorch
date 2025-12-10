CMAKE_PREFIX_PATH
-----------------

.. include:: include/ENV_VAR.rst

The ``CMAKE_PREFIX_PATH`` environment variable may be set to a list of
directories specifying installation *prefixes* to be searched by the
:command:`find_package`, :command:`find_program`, :command:`find_library`,
:command:`find_file`, and :command:`find_path` commands.  Each command will
add appropriate subdirectories (like ``bin``, ``lib``, or ``include``)
as specified in its own documentation.

This variable may hold a single prefix or a list of prefixes separated
by ``:`` on UNIX or ``;`` on Windows (the same as the ``PATH`` environment
variable convention on those platforms).

See also the :variable:`CMAKE_PREFIX_PATH` CMake variable.
