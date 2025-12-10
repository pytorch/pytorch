CMAKE_FIND_DEBUG_MODE
---------------------

.. versionadded:: 3.17

Print extra find call information for the following commands to standard
error:

* :command:`find_program`
* :command:`find_library`
* :command:`find_file`
* :command:`find_path`
* :command:`find_package`

Output is designed for human consumption and not for parsing.
Enabling this variable is equivalent to using :option:`cmake --debug-find`
with the added ability to enable debugging for a subset of find calls.

.. code-block:: cmake

  set(CMAKE_FIND_DEBUG_MODE TRUE)
  find_program(...)
  set(CMAKE_FIND_DEBUG_MODE FALSE)

Default is unset.
