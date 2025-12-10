CMAKE_<LANG>_ANDROID_TOOLCHAIN_MACHINE
--------------------------------------

.. versionadded:: 3.7.1

When :ref:`Cross Compiling for Android` this variable contains the
toolchain binutils machine name (e.g. ``gcc -dumpmachine``).  The
binutils typically have a ``<machine>-`` prefix on their name.

See also :variable:`CMAKE_<LANG>_ANDROID_TOOLCHAIN_PREFIX`
and :variable:`CMAKE_<LANG>_ANDROID_TOOLCHAIN_SUFFIX`.
