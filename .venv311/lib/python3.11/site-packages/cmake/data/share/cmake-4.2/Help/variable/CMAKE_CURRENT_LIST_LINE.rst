CMAKE_CURRENT_LIST_LINE
-----------------------

The line number of the current file being processed.

This is the line number of the file currently being processed by
cmake.

If CMake is currently processing deferred calls scheduled by
the :command:`cmake_language(DEFER)` command, this variable
evaluates to ``DEFERRED`` instead of a specific line number.

See also :variable:`CMAKE_CURRENT_FUNCTION_LIST_LINE`.
