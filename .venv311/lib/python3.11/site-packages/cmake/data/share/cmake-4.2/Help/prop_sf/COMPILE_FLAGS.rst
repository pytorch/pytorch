COMPILE_FLAGS
-------------

Additional flags to be added when compiling this source file.

The ``COMPILE_FLAGS`` property, managed as a string, sets additional compiler
flags used that will be added to the list of compile flags when this source
file builds.  The flags will be added after target-wide flags.

Use :prop_sf:`COMPILE_DEFINITIONS` to pass additional preprocessor definitions.

Contents of ``COMPILE_FLAGS`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.  However, :generator:`Xcode`
does not support per-config per-source settings, so expressions
that depend on the build configuration are not allowed with that
generator.

.. note::

  This property has been superseded by the :prop_sf:`COMPILE_OPTIONS` property.
