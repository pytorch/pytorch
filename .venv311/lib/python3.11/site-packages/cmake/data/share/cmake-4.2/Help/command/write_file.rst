write_file
----------

.. deprecated:: 3.0

  Use the :command:`file(WRITE)` command instead.

.. code-block:: cmake

  write_file(filename "message to write"... [APPEND])

The first argument is the file name, the rest of the arguments are
messages to write.  If the argument ``APPEND`` is specified, then the
message will be appended.

NOTE 1: :command:`file(WRITE)`  and :command:`file(APPEND)`  do exactly
the same as this one but add some more functionality.

NOTE 2: When using ``write_file`` the produced file cannot be used as an
input to CMake (CONFIGURE_FILE, source file ...) because it will lead
to an infinite loop.  Use :command:`configure_file` if you want to
generate input files to CMake.
