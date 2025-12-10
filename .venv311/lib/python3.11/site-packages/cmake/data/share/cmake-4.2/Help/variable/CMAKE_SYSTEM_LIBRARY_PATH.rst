CMAKE_SYSTEM_LIBRARY_PATH
-------------------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories specifying a search path
for the :command:`find_library` command.  By default this contains the
standard directories for the current system.  It is *not* intended to be
modified by the project; use :variable:`CMAKE_LIBRARY_PATH` for this.
See also :variable:`CMAKE_SYSTEM_PREFIX_PATH`.
