AUTOGEN_COMMAND_LINE_LENGTH_MAX
-------------------------------

.. versionadded:: 3.29

Command line length limit for autogen targets, i.e. ``moc`` or ``uic``,
that triggers the use of response files on Windows instead of passing all
arguments to the command line.

- An empty (or unset) value sets the limit to 32000
- A positive non zero integer value sets the exact command line length
  limit.

By default ``AUTOGEN_COMMAND_LINE_LENGTH_MAX`` is initialized from
:variable:`CMAKE_AUTOGEN_COMMAND_LINE_LENGTH_MAX`.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
