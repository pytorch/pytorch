INTERPROCEDURAL_OPTIMIZATION_<CONFIG>
-------------------------------------

Per-configuration interprocedural optimization for a target.

This is a per-configuration version of :prop_tgt:`INTERPROCEDURAL_OPTIMIZATION`.
If set, this property overrides the generic property for the named
configuration.

This property is initialized by the
:variable:`CMAKE_INTERPROCEDURAL_OPTIMIZATION_<CONFIG>` variable if it is set
when a target is created.

See Also
^^^^^^^^

* The :module:`CheckIPOSupported` module to check whether the compiler
  supports interprocedural optimization before enabling this target property.
