VERSION
-------

Version number of a shared library target.

For shared libraries ``VERSION`` and :prop_tgt:`SOVERSION` can be used
to specify the build version and ABI version respectively.  When building or
installing appropriate symlinks are created if the platform supports
symlinks and the linker supports so-names.  If only one of both is
specified the missing is assumed to have the same version number.  For
executables ``VERSION`` can be used to specify the build version.  When
building or installing appropriate symlinks are created if the
platform supports symlinks.

.. include:: include/VERSION_SOVERSION_EXAMPLE.rst

Windows Versions
^^^^^^^^^^^^^^^^

For shared libraries and executables on Windows the ``VERSION``
attribute is parsed to extract a ``<major>.<minor>`` version number.
These numbers are used as the image version of the binary.

Mach-O Versions
^^^^^^^^^^^^^^^

For shared libraries and executables on Mach-O systems (e.g. macOS, iOS),
the :prop_tgt:`SOVERSION` property corresponds to the *compatibility version*
and ``VERSION`` corresponds to the *current version* (unless Mach-O specific
overrides are provided, as discussed below).
See the :prop_tgt:`FRAMEWORK` target property for an example.

For shared libraries, the :prop_tgt:`MACHO_COMPATIBILITY_VERSION` and
:prop_tgt:`MACHO_CURRENT_VERSION` properties can be used to
override the *compatibility version* and *current version* respectively.
Note that :prop_tgt:`SOVERSION` will still be used to form the
``install_name`` and both :prop_tgt:`SOVERSION` and ``VERSION`` may also
affect the file and symlink names.

Versions of Mach-O binaries may be checked with the ``otool -L <binary>``
command.
