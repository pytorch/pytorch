CMAKE_OBJDUMP
-------------

Path to the ``objdump`` executable on the host system.  This tool, typically
part of the Binutils collection on Unix-like systems, provides information
about compiled object files.

This cache variable may be populated by CMake when project languages are
enabled using the :command:`project` or :command:`enable_language` commands.

See Also
^^^^^^^^

* The :command:`file(GET_RUNTIME_DEPENDENCIES)` command provides a more general
  way to get information from runtime binaries.
* The :variable:`CPACK_OBJDUMP_EXECUTABLE` variable.
