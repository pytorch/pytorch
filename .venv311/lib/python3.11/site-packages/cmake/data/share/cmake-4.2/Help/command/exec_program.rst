exec_program
------------

.. versionchanged:: 3.28
  This command is available only if policy :policy:`CMP0153` is not set to ``NEW``.
  Port projects to the :command:`execute_process` command.

.. deprecated:: 3.0

  Use the :command:`execute_process` command instead.

Runs an executable program during the processing of a CMake file or script:

.. code-block:: cmake

  exec_program(
    <executable>
    [<working-dir>]
    [ARGS <arguments-to-executable>...]
    [OUTPUT_VARIABLE <var>]
    [RETURN_VALUE <var>]
  )

The ``<executable>`` is run in the optionally specified directory
``<working-dir>``.  The
executable can include arguments if it is double quoted, but it is
better to use the optional ``ARGS`` argument to specify arguments to the
executable program.  This is because CMake will then be able to escape spaces in
the executable path.  An optional argument ``OUTPUT_VARIABLE`` specifies a
variable in which to store the output.  To capture the return value of
the execution, provide a ``RETURN_VALUE``.  If ``OUTPUT_VARIABLE`` is
specified, then no output will go to the stdout/stderr of the console
running CMake.

Examples
^^^^^^^^

Example of the legacy ``exec_program()`` command used in earlier versions of
CMake:

.. code-block:: cmake

  exec_program(
    some_command
    ${dir}
    ARGS arg_1 arg_2 args "\"<quoted-arg>\""
    OUTPUT_VARIABLE output
    RETURN_VALUE result
  )

A direct equivalent replacement of the previous example using the
:command:`execute_process` command in new code:

.. code-block:: cmake

  execute_process(
    COMMAND some_command arg_1 arg_2 args "<quoted-arg>"
    WORKING_DIRECTORY ${dir}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
