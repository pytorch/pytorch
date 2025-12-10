COMPILE_DEFINITIONS_<CONFIG>
----------------------------

Ignored.  See CMake Policy :policy:`CMP0043`.

Per-configuration preprocessor definitions in a directory.

This is the configuration-specific version of :prop_dir:`COMPILE_DEFINITIONS`
where ``<CONFIG>`` is an upper-case name (ex. ``COMPILE_DEFINITIONS_DEBUG``).

This property will be initialized in each directory by its value in
the directory's parent.

Contents of ``COMPILE_DEFINITIONS_<CONFIG>`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  See the :manual:`cmake-buildsystem(7)`
manual for more on defining buildsystem properties.

Generator expressions should be preferred instead of setting this property.
