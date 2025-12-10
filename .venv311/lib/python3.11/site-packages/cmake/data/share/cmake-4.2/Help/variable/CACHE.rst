CACHE
-----

.. versionadded:: 3.13

Operator to read cache variables.

Use the syntax ``$CACHE{VAR}`` to read cache entry ``VAR``.
See the :ref:`cmake-language(7) variables <CMake Language Variables>`
documentation for more complete documentation of the interaction of
normal variables and cache entries.

When evaluating :ref:`Variable References` of the form ``${VAR}``,
CMake first searches for a normal variable with that name, and if not
found CMake will search for a cache entry with that name.
The ``$CACHE{VAR}`` syntax can be used to do direct cache lookup and
ignore any existing normal variable.

See the :command:`set` and :command:`unset` commands to see how to
write or remove cache variables.
