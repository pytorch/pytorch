EXCLUDE_FROM_ALL
----------------

Set this target property to a true (or false) value to exclude (or include)
the target from the "all" target of the containing directory and its
ancestors.  If excluded, running e.g. ``make`` in the containing directory
or its ancestors will not build the target by default.

If this target property is not set then the target will be included in
the "all" target of the containing directory.  Furthermore, it will be
included in the "all" target of its ancestor directories unless the
:prop_dir:`EXCLUDE_FROM_ALL` directory property is set.

With ``EXCLUDE_FROM_ALL`` set to false or not set at all, the target
will be brought up to date as part of doing a ``make install`` or its
equivalent for the CMake generator being used.

If a target has ``EXCLUDE_FROM_ALL`` set to true, it may still be listed
in an :command:`install(TARGETS)` command, but the user is responsible for
ensuring that the target's build artifacts are not missing or outdated when
an install is performed.

This property may use "generator expressions" with the syntax ``$<...>``. See
the :manual:`cmake-generator-expressions(7)` manual for available expressions.

Only the "Ninja Multi-Config" generator supports a property value that varies by
configuration.  For all other generators the value of this property must be the
same for all configurations.

See Also
^^^^^^^^

* To exclude targets from the whole directory subtree, see the
  :prop_dir:`EXCLUDE_FROM_ALL` directory property.

* To exclude targets from the Visual Studio solution build, use
  :prop_tgt:`EXCLUDE_FROM_DEFAULT_BUILD`.
