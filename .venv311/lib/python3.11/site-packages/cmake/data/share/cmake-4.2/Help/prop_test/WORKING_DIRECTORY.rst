WORKING_DIRECTORY
-----------------

The directory from which the test executable will be called.

If this is not set, the test will be run with the working directory set to the
binary directory associated with where the test was created (i.e. the
:variable:`CMAKE_CURRENT_BINARY_DIR` for where :command:`add_test` was
called).
