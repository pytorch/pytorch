IMPORTED_IMPLIB
---------------

Full path to the import library for an ``IMPORTED`` target.

This property may be set:

* On DLL platforms, to the location of the ``.lib`` part of the DLL.
* .. versionadded:: 3.16
    On AIX, to an import file (e.g. ``.imp``) created for executables that
    export symbols (see the :prop_tgt:`ENABLE_EXPORTS` target property).
* .. versionadded:: 3.27
    On Apple platforms, to an import file (e.g. ``.tbd``) created for shared
    libraries or frameworks (see the :prop_tgt:`ENABLE_EXPORTS` target
    property).  For frameworks, this is the location of the ``.tbd`` file
    symlink just inside the framework folder.
* .. versionadded:: 3.28
    On non-DLL platforms, to the location of a shared library.
    When set without also specifying an :prop_tgt:`IMPORTED_LOCATION`,
    the library is considered to be a stub, and its location will not
    be added as a runtime search path to dependents that link it.

.. versionchanged:: 3.28
  If an imported target is an Apple framework or XCFramework, the preferred
  arrangement is to set :prop_tgt:`IMPORTED_LOCATION` to the ``.framework``
  or ``.xcframework`` directory.  CMake will then find the relevant ``.tbd``
  file inside that framework or XCFramework automatically without requiring
  ``IMPORTED_IMPLIB`` to be set.

The ``IMPORTED_IMPLIB`` target property may be overridden for a
given configuration ``<CONFIG>`` by the configuration-specific
:prop_tgt:`IMPORTED_IMPLIB_<CONFIG>` target property.  Furthermore,
the :prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` target property may be
used to map between a project's configurations and those of an imported
target.  If none of these is set then the name of any other configuration
listed in the :prop_tgt:`IMPORTED_CONFIGURATIONS` target property may be
selected and its :prop_tgt:`IMPORTED_IMPLIB_<CONFIG>` value used.

This property is ignored for non-imported targets.
