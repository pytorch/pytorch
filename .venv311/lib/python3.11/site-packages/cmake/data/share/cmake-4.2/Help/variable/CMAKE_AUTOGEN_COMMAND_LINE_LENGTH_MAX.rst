CMAKE_AUTOGEN_COMMAND_LINE_LENGTH_MAX
-------------------------------------

.. versionadded:: 3.29

Command line length limit for autogen targets, i.e. ``moc`` or ``uic``,
that triggers the use of response files on Windows instead of passing all
arguments to the command line.

By default ``CMAKE_AUTOGEN_COMMAND_LINE_LENGTH_MAX`` is unset.
