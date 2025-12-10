CMAKE_MAXIMUM_RECURSION_DEPTH
-----------------------------

.. versionadded:: 3.14

Maximum recursion depth for CMake scripts. It is intended to be set on the
command line with ``-DCMAKE_MAXIMUM_RECURSION_DEPTH=<x>``, or within
``CMakeLists.txt`` by projects that require a large recursion depth. Projects
that set this variable should provide the user with a way to override it. For
example:

.. code-block:: cmake

  # About to perform deeply recursive actions
  if(NOT CMAKE_MAXIMUM_RECURSION_DEPTH)
    set(CMAKE_MAXIMUM_RECURSION_DEPTH 2000)
  endif()

If it is not set, or is set to a non-integer value, a sensible default limit is
used. If the recursion limit is reached, the script terminates immediately with
a fatal error.

Calling any of the following commands increases the recursion depth:

* :command:`include`
* :command:`find_package`
* :command:`add_subdirectory`
* :command:`try_compile`
* :command:`ctest_read_custom_files`
* :command:`ctest_run_script` (unless ``NEW_PROCESS`` is specified)
* User-defined :command:`function`'s and :command:`macro`'s (note that
  :command:`function` and :command:`macro` themselves don't increase recursion
  depth)
* Reading or writing variables that are being watched by a
  :command:`variable_watch`

See also the :envvar:`CMAKE_MAXIMUM_RECURSION_DEPTH` environment variable.
