COMPILE_DEFINITIONS_<CONFIG>
----------------------------

Ignored.  See CMake Policy :policy:`CMP0043`.

Per-configuration preprocessor definitions on a target.

This is the configuration-specific version of :prop_tgt:`COMPILE_DEFINITIONS`
where ``<CONFIG>`` is an upper-case name (ex. ``COMPILE_DEFINITIONS_DEBUG``).

Contents of ``COMPILE_DEFINITIONS_<CONFIG>`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.

Generator expressions should be preferred instead of setting this property.
