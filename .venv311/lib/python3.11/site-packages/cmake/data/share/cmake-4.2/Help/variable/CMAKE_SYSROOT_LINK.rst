CMAKE_SYSROOT_LINK
------------------

.. versionadded:: 3.9

Path to pass to the compiler in the ``--sysroot`` flag when linking.  This is
the same as :variable:`CMAKE_SYSROOT` but is used only for linking and not
compiling sources.

This variable may only be set in a toolchain file specified by
the :variable:`CMAKE_TOOLCHAIN_FILE` variable.
