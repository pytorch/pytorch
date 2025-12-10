CMAKE_FIND_DEBUG_MODE_NO_IMPLICIT_CONFIGURE_LOG
-----------------------------------------------

.. versionadded:: 4.1

The following commands will report configure log events when they experience a
transition between found and not-found states or when the result is first
defined:

* :command:`find_program`
* :command:`find_library`
* :command:`find_file`
* :command:`find_path`
* :command:`find_package`

The ``CMAKE_FIND_DEBUG_MODE_NO_IMPLICIT_CONFIGURE_LOG`` boolean variable
suppresses these implicit events from the configure log when set to a true
value.

.. code-block:: cmake

  set(CMAKE_FIND_DEBUG_MODE_NO_IMPLICIT_CONFIGURE_LOG TRUE)
  find_program(...)
  set(CMAKE_FIND_DEBUG_MODE_NO_IMPLICIT_CONFIGURE_LOG FALSE)

Default is unset.
