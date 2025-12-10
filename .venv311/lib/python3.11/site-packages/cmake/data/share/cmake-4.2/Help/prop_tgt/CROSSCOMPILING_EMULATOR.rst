CROSSCOMPILING_EMULATOR
-----------------------

.. versionadded:: 3.3

Use the given emulator to run executables created when crosscompiling.
This command will be added as a prefix to :command:`add_test`,
:command:`add_custom_command`, and :command:`add_custom_target` commands
for built target system executables.

.. versionadded:: 3.15
  If this property contains a :ref:`semicolon-separated list <CMake Language
  Lists>`, then the first value is the command and remaining values are its
  arguments.

.. versionadded:: 3.29
  Contents of ``CROSSCOMPILING_EMULATOR`` may use
  :manual:`generator expressions <cmake-generator-expressions(7)>`.

This property is initialized by the value of the
:variable:`CMAKE_CROSSCOMPILING_EMULATOR` variable if it is set when a target
is created.

This property is not supported when using the old form of :command:`add_test`
(i.e. without the ``NAME`` and ``COMMAND`` keywords).
