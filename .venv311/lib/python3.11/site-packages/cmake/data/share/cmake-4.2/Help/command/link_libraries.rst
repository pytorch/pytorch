link_libraries
--------------

Link libraries to all targets added later.

.. code-block:: cmake

  link_libraries([item1 [item2 [...]]]
                 [[debug|optimized|general] <item>] ...)

Specify libraries or flags to use when linking any targets created later in
the current directory or below by commands such as :command:`add_executable`
or :command:`add_library`.  See the :command:`target_link_libraries` command
for meaning of arguments.

.. note::
  The :command:`target_link_libraries` command should be preferred whenever
  possible.  Library dependencies are chained automatically, so directory-wide
  specification of link libraries is rarely needed.
