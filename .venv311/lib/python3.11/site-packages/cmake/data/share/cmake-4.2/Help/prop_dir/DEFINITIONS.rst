DEFINITIONS
-----------

For CMake 2.4 compatibility only.  Use :prop_dir:`COMPILE_DEFINITIONS`
instead.

This read-only property specifies the list of flags given so far to
the :command:`add_definitions` command.  It is intended for debugging
purposes.  Use the :prop_dir:`COMPILE_DEFINITIONS` directory property
instead.

This built-in read-only property does not exist if policy
:policy:`CMP0059` is set to ``NEW``.
