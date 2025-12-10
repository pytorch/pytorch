CMAKE_TEST_LAUNCHER
-------------------

.. versionadded:: 3.29

This variable is used to initialize the :prop_tgt:`TEST_LAUNCHER` target
property of executable targets as they are created.  It is used to specify
a launcher for running tests, added by the :command:`add_test` command,
that run an executable target.

If this variable contains a :ref:`semicolon-separated list <CMake Language
Lists>`, then the first value is the command and remaining values are its
arguments.

This variable can be initialized via an
:envvar:`CMAKE_TEST_LAUNCHER` environment variable.
