CMAKE_PROGRAM_PATH
------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying a search path
for the :command:`find_program` command.  By default it is empty, it is
intended to be set by the project.

There is also an environment variable :envvar:`CMAKE_PROGRAM_PATH`, which is used
as an additional list of search directories.

See also :variable:`CMAKE_SYSTEM_PROGRAM_PATH` and :variable:`CMAKE_PREFIX_PATH`.
