CMAKE_BINARY_DIR
----------------

The path to the top level of the build tree.

This is the full path to the top level of the current CMake build
tree.  For an in-source build, this would be the same as
:variable:`CMAKE_SOURCE_DIR`.

When run in :option:`cmake -P` script mode, CMake sets the variables
``CMAKE_BINARY_DIR``, :variable:`CMAKE_SOURCE_DIR`,
:variable:`CMAKE_CURRENT_BINARY_DIR` and
:variable:`CMAKE_CURRENT_SOURCE_DIR` to the current working directory.

Modifying ``CMAKE_BINARY_DIR`` has undefined behavior.
