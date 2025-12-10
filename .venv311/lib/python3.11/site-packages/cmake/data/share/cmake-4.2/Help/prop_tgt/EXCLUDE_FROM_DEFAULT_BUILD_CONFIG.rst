EXCLUDE_FROM_DEFAULT_BUILD_<CONFIG>
-----------------------------------

Per-configuration version of target exclusion from ``Build Solution``.

This is the configuration-specific version of
:prop_tgt:`EXCLUDE_FROM_DEFAULT_BUILD`.  If the generic
:prop_tgt:`EXCLUDE_FROM_DEFAULT_BUILD` is also set on a target,
``EXCLUDE_FROM_DEFAULT_BUILD_<CONFIG>`` takes
precedence in configurations for which it has a value.
