CMAKE_CROSSCOMPILING_EMULATOR
-----------------------------

.. versionadded:: 3.3

This variable is only used when :variable:`CMAKE_CROSSCOMPILING` is on. It
should point to a command on the host system that can run executable built
for the target system.

.. versionadded:: 3.15
  If this variable contains a :ref:`semicolon-separated list <CMake Language
  Lists>`, then the first value is the command and remaining values are its
  arguments.

.. versionadded:: 3.28
  This variable can be initialized via an
  :envvar:`CMAKE_CROSSCOMPILING_EMULATOR` environment variable.

The command will be used to run :command:`try_run` generated executables,
which avoids manual population of the ``TryRunResults.cmake`` file.

This variable is also used as the default value for the
:prop_tgt:`CROSSCOMPILING_EMULATOR` target property of executables.  However,
while :manual:`generator expressions <cmake-generator-expressions(7)>` are
supported by the target property (since CMake 3.29), they are *not* supported
by this variable's :command:`try_run` functionality.
