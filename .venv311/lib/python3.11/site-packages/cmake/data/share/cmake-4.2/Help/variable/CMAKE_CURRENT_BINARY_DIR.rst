CMAKE_CURRENT_BINARY_DIR
------------------------

The path to the binary directory currently being processed.

This is the full path to the build directory that is currently being
processed by cmake.  Each directory added by :command:`add_subdirectory` will
create a binary directory in the build tree, and as it is being
processed this variable will be set.  For in-source builds this is the
current source directory being processed.

When run in :option:`cmake -P` script mode, CMake sets the variables
:variable:`CMAKE_BINARY_DIR`, :variable:`CMAKE_SOURCE_DIR`,
``CMAKE_CURRENT_BINARY_DIR`` and
:variable:`CMAKE_CURRENT_SOURCE_DIR` to the current working directory.

Modifying ``CMAKE_CURRENT_BINARY_DIR`` has undefined behavior.
