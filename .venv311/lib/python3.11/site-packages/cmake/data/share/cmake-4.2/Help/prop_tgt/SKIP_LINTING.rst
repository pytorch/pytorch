SKIP_LINTING
------------

.. versionadded:: 4.2

Exclude all sources of a target from running configured linting tools.

When this boolean property is enabled on a target, C/C++ linting tools enabled
for that target (e.g. :prop_tgt:`<LANG>_CPPLINT`, :prop_tgt:`<LANG>_CLANG_TIDY`,
:prop_tgt:`<LANG>_CPPCHECK`, :prop_tgt:`<LANG>_ICSTAT` and
:prop_tgt:`<LANG>_INCLUDE_WHAT_YOU_USE`) will not be invoked for source files
compiled by the target.  If the :prop_sf:`SKIP_LINTING` source-file property
is set on a specific source, it takes precedence over this target-wide property.

This is a convenience alternative to setting the :prop_sf:`SKIP_LINTING`
source file property individually on each source.  If either the target's
:prop_tgt:`SKIP_LINTING` or a source’s :prop_sf:`SKIP_LINTING` is enabled,
that source will be excluded from linting.

The property has no effect on targets that do not have sources.

See Also
^^^^^^^^

* :prop_sf:`SKIP_LINTING` source file property
