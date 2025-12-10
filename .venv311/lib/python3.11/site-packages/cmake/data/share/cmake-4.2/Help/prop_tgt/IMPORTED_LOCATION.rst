IMPORTED_LOCATION
-----------------

Full path to the main file on disk for an ``IMPORTED`` target.

Set this to the location of an ``IMPORTED`` target file on disk.  For
executables this is the location of the executable file.  For ``STATIC``
libraries and modules this is the location of the library or module.
For ``SHARED`` libraries on non-DLL platforms this is the location of the
shared library.  For application bundles on macOS this is the location of
the executable file inside ``Contents/MacOS`` within the bundle folder.
For frameworks on macOS this is the location of the
library file symlink just inside the framework folder.  For DLLs this
is the location of the ``.dll`` part of the library.  For ``UNKNOWN``
libraries this is the location of the file to be linked.  Ignored for
non-imported targets.

.. versionadded:: 3.28

  For ordinary frameworks on Apple platforms, this may be the location of the
  ``.framework`` folder itself.  For XCFrameworks, it may be the location of
  the ``.xcframework`` folder, in which case any target that links against it
  will get the selected library's ``Headers`` directory as a usage requirement.

The ``IMPORTED_LOCATION`` target property may be overridden for a
given configuration ``<CONFIG>`` by the configuration-specific
:prop_tgt:`IMPORTED_LOCATION_<CONFIG>` target property.  Furthermore,
the :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` target property may be
used to map between a project's configurations and those of an imported
target.  If none of these is set then the name of any other configuration
listed in the :prop_tgt:`IMPORTED_CONFIGURATIONS` target property may be
selected and its :prop_tgt:`IMPORTED_LOCATION_<CONFIG>` value used.

To get the location of an imported target read one of the :prop_tgt:`LOCATION`
or :prop_tgt:`LOCATION_<CONFIG>` properties.

For platforms with import libraries (e.g. Windows, AIX or Apple) see also
:prop_tgt:`IMPORTED_IMPLIB`.
