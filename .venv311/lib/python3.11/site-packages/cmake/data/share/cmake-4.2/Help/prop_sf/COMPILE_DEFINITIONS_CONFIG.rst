COMPILE_DEFINITIONS_<CONFIG>
----------------------------

.. deprecated:: 3.0

  Prefer the :prop_sf:`COMPILE_DEFINITIONS` source-file property with
  :manual:`generator expressions <cmake-generator-expressions(7)>`.

Per-configuration preprocessor definitions on a source file.

This is the configuration-specific version of :prop_sf:`COMPILE_DEFINITIONS`.

Note that :generator:`Xcode` does not support per-configuration source
file flags so this property will be ignored by the :generator:`Xcode` generator.
