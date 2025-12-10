CMAKE_SYSTEM_INCLUDE_PATH
-------------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying a search path
for the :command:`find_file` and :command:`find_path` commands.  By default
this contains the standard directories for the current system.  It is *not*
intended to be modified by the project; use :variable:`CMAKE_INCLUDE_PATH` for
this.  See also :variable:`CMAKE_SYSTEM_PREFIX_PATH`.
