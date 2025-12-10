MACHO_COMPATIBILITY_VERSION
---------------------------

.. versionadded:: 3.17

What compatibility version number is this target for Mach-O binaries.

For shared libraries on Mach-O systems (e.g. macOS, iOS)
the ``MACHO_COMPATIBILITY_VERSION`` property corresponds to the
*compatibility version* and :prop_tgt:`MACHO_CURRENT_VERSION` corresponds to
the *current version*.  These are both embedded in the shared library binary
and can be checked with the ``otool -L <binary>`` command.

It should be noted that the :prop_tgt:`MACHO_CURRENT_VERSION` and
``MACHO_COMPATIBILITY_VERSION`` properties do not affect the file
names or version-related symlinks that CMake generates for the library.
The :prop_tgt:`VERSION` and :prop_tgt:`SOVERSION` target properties still
control the file and symlink names.  The ``install_name`` is also still
controlled by :prop_tgt:`SOVERSION`.

When :prop_tgt:`MACHO_CURRENT_VERSION` and ``MACHO_COMPATIBILITY_VERSION``
are not given, :prop_tgt:`VERSION` and :prop_tgt:`SOVERSION` are used for
the version details to be embedded in the binaries respectively.
The :prop_tgt:`MACHO_CURRENT_VERSION` and ``MACHO_COMPATIBILITY_VERSION``
properties only need to be given if the project needs to decouple the file
and symlink naming from the version details embedded in the binaries
(e.g. to match libtool conventions).
