separate_arguments
------------------

Parse command-line arguments into a semicolon-separated list.

.. code-block:: cmake

  separate_arguments(<variable> <mode> [PROGRAM [SEPARATE_ARGS]] <args>)

Parses a space-separated string ``<args>`` into a list of items,
and stores this list in semicolon-separated standard form in ``<variable>``.

This function is intended for parsing command-line arguments.
The entire command line must be passed as one string in the
argument ``<args>``.

The exact parsing rules depend on the operating system.
They are specified by the ``<mode>`` argument which must
be one of the following keywords:

``UNIX_COMMAND``
  Arguments are separated by unquoted whitespace.
  Both single-quote and double-quote pairs are respected.
  A backslash escapes the next literal character (``\"`` is ``"``);
  there are no special escapes (``\n`` is just ``n``).

``WINDOWS_COMMAND``
  A Windows command-line is parsed using the same
  syntax the runtime library uses to construct argv at startup.  It
  separates arguments by whitespace that is not double-quoted.
  Backslashes are literal unless they precede double-quotes.  See the
  MSDN article `Parsing C Command-Line Arguments`_ for details.

``NATIVE_COMMAND``
  .. versionadded:: 3.9

  Proceeds as in ``WINDOWS_COMMAND`` mode if the host system is Windows.
  Otherwise proceeds as in ``UNIX_COMMAND`` mode.

``PROGRAM``
  .. versionadded:: 3.19

  The first item in ``<args>`` is assumed to be an executable and will be
  searched in the system search path or left as a full path. If not found,
  ``<variable>`` will be empty. Otherwise, ``<variable>`` is a list of 2
  elements:

  0. Absolute path of the program
  1. Any command-line arguments present in ``<args>`` as a string

  For example:

  .. code-block:: cmake

    separate_arguments (out UNIX_COMMAND PROGRAM "cc -c main.c")

  * First element of the list: ``/path/to/cc``
  * Second element of the list: ``" -c main.c"``

``SEPARATE_ARGS``
  When this sub-option of ``PROGRAM`` option is specified, command-line
  arguments will be split as well and stored in ``<variable>``.

  For example:

  .. code-block:: cmake

    separate_arguments (out UNIX_COMMAND PROGRAM SEPARATE_ARGS "cc -c main.c")

  The contents of ``out`` will be: ``/path/to/cc;-c;main.c``

.. _`Parsing C Command-Line Arguments`: https://learn.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments

.. code-block:: cmake

  separate_arguments(<var>)

Convert the value of ``<var>`` to a semi-colon separated list.  All
spaces are replaced with ';'.  This helps with generating command
lines.
