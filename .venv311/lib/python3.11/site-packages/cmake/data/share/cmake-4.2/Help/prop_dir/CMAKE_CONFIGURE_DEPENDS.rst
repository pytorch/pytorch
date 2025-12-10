CMAKE_CONFIGURE_DEPENDS
-----------------------

Tell CMake about additional input files to the configuration process.
If any named file is modified the build system will re-run CMake to
re-configure the file and generate the build system again.

Specify files as a semicolon-separated list of paths.  Relative paths
are interpreted as relative to the current source directory.
