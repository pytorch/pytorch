CMAKE_INCLUDE_CURRENT_DIR
-------------------------

Automatically add the current source and build directories to the include path.

If this variable is enabled, CMake automatically adds
:variable:`CMAKE_CURRENT_SOURCE_DIR` and :variable:`CMAKE_CURRENT_BINARY_DIR`
to the include path for each directory.  These additional include
directories do not propagate down to subdirectories.  This is useful
mainly for out-of-source builds, where files generated into the build
tree are included by files located in the source tree.

By default ``CMAKE_INCLUDE_CURRENT_DIR`` is ``OFF``.
