CMAKE_SYSROOT_COMPILE
---------------------

.. versionadded:: 3.9

Path to pass to the compiler in the ``--sysroot`` flag when compiling source
files.  This is the same as :variable:`CMAKE_SYSROOT` but is used only for
compiling sources and not linking.

This variable may only be set in a toolchain file specified by
the :variable:`CMAKE_TOOLCHAIN_FILE` variable.
