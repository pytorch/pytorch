CMAKE_PREFIX_PATH
-----------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying installation
*prefixes* to be searched by the :command:`find_package`,
:command:`find_program`, :command:`find_library`, :command:`find_file`, and
:command:`find_path` commands.  Each command will add appropriate
subdirectories (like ``bin``, ``lib``, or ``include``) as specified in its own
documentation.

By default this is empty.  It is intended to be set by the project.

There is also an environment variable :envvar:`CMAKE_PREFIX_PATH`, which is used
as an additional list of search prefixes.

See also :variable:`CMAKE_SYSTEM_PREFIX_PATH`, :variable:`CMAKE_INCLUDE_PATH`,
:variable:`CMAKE_LIBRARY_PATH`, :variable:`CMAKE_PROGRAM_PATH`, and
:variable:`CMAKE_IGNORE_PATH`.
