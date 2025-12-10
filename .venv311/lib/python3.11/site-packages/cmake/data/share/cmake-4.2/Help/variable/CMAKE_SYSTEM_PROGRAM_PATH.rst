CMAKE_SYSTEM_PROGRAM_PATH
-------------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying a search path
for the :command:`find_program` command.  By default this contains the
standard directories for the current system.  It is *not* intended to be
modified by the project; use :variable:`CMAKE_PROGRAM_PATH` for this.
See also :variable:`CMAKE_SYSTEM_PREFIX_PATH`.
