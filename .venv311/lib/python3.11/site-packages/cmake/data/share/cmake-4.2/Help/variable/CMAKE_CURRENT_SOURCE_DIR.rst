CMAKE_CURRENT_SOURCE_DIR
------------------------

The path to the source directory currently being processed.

This is the full path to the source directory that is currently being
processed by cmake.

When run in :option:`cmake -P` script mode, CMake sets the variables
:variable:`CMAKE_BINARY_DIR`, :variable:`CMAKE_SOURCE_DIR`,
:variable:`CMAKE_CURRENT_BINARY_DIR` and
``CMAKE_CURRENT_SOURCE_DIR`` to the current working directory.

Modifying ``CMAKE_CURRENT_SOURCE_DIR`` has undefined behavior.
