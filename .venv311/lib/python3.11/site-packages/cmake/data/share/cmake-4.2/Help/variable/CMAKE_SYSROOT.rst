CMAKE_SYSROOT
-------------

Path to pass to the compiler in the ``--sysroot`` flag.

The ``CMAKE_SYSROOT`` content is passed to the compiler in the ``--sysroot``
flag, if supported.  The path is also stripped from the ``RPATH``/``RUNPATH``
if necessary on installation.  The ``CMAKE_SYSROOT`` is also used to prefix
paths searched by the ``find_*`` commands.

This variable may only be set in a toolchain file specified by
the :variable:`CMAKE_TOOLCHAIN_FILE` variable.

See also the :variable:`CMAKE_SYSROOT_COMPILE` and
:variable:`CMAKE_SYSROOT_LINK` variables.
