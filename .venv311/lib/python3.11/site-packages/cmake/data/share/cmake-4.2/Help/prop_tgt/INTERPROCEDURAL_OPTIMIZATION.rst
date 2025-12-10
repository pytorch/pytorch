INTERPROCEDURAL_OPTIMIZATION
----------------------------

Enable interprocedural optimization for a target.

If set to true, enables interprocedural optimizations if they are
known :module:`to be supported <CheckIPOSupported>` by the compiler. Depending
on value of policy :policy:`CMP0069`, the error will be reported or ignored,
if interprocedural optimization is enabled but not supported.

This property is initialized by the
:variable:`CMAKE_INTERPROCEDURAL_OPTIMIZATION` variable if it is set when a
target is created.

There is also the per-configuration :prop_tgt:`INTERPROCEDURAL_OPTIMIZATION_<CONFIG>`
target property, which overrides :prop_tgt:`INTERPROCEDURAL_OPTIMIZATION`
if it is set.

See Also
^^^^^^^^

* The :module:`CheckIPOSupported` module to check whether the compiler
  supports interprocedural optimization before enabling this target property.
