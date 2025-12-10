TEST_LAUNCHER
-------------

.. versionadded:: 3.29

Use the given launcher to run executables.
This command will be added as a prefix to :command:`add_test` commands
for build target system executables and is meant to be run on the host
machine.

It effectively acts as a run script for tests in a similar way
to how :variable:`CMAKE_<LANG>_COMPILER_LAUNCHER` works for compilation.

If this property contains a :ref:`semicolon-separated list <CMake Language
Lists>`, then the first value is the command and remaining values are its
arguments.

Contents of ``TEST_LAUNCHER`` may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is initialized by the value of the
:variable:`CMAKE_TEST_LAUNCHER` variable if it is set when a target
is created.

This property is not supported when using the old form of :command:`add_test`
(i.e. without the ``NAME`` and ``COMMAND`` keywords).
