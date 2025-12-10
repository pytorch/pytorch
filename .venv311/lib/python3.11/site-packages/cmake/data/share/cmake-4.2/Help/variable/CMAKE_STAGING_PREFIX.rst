CMAKE_STAGING_PREFIX
--------------------

This variable may be set to a path to install to when cross-compiling. This can
be useful if the path in :variable:`CMAKE_SYSROOT` is read-only, or otherwise
should remain pristine.

The ``CMAKE_STAGING_PREFIX`` location is also used as a search prefix
by the ``find_*`` commands. This can be controlled by setting the
:variable:`CMAKE_FIND_NO_INSTALL_PREFIX` variable.

If any ``RPATH``/``RUNPATH`` entries passed to the linker contain the
``CMAKE_STAGING_PREFIX``, the matching path fragments are replaced
with the :variable:`CMAKE_INSTALL_PREFIX`.
