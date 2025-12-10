IMPORTED_CONFIGURATIONS
-----------------------

Configurations provided for an :ref:`imported target <Imported targets>`.

Set this to the list of configuration names available for an imported
target.  For each configuration named, the imported target's artifacts
must be specified in other target properties:

* :prop_tgt:`IMPORTED_LOCATION_<CONFIG>`, or
* :prop_tgt:`IMPORTED_IMPLIB_<CONFIG>` (on DLL platforms, on AIX for
  :ref:`Executables` or on Apple for :ref:`Shared Libraries`), or
* :prop_tgt:`IMPORTED_OBJECTS_<CONFIG>` (for :ref:`Object Libraries`), or
* :prop_tgt:`IMPORTED_LIBNAME_<CONFIG>` (for :ref:`Interface Libraries`).

The configuration names correspond to those defined in the project from
which the target is imported.  If the importing project uses a different
set of configurations, the names may be mapped using the
:prop_tgt:`MAP_IMPORTED_CONFIG_<CONFIG>` target property.

The ``IMPORTED_CONFIGURATIONS`` property is ignored for non-imported targets.
