PCH_WARN_INVALID
----------------

.. versionadded:: 3.18

When this property is set to true, the precompile header compiler options
will contain a compiler flag which should warn about invalid precompiled
headers e.g. ``-Winvalid-pch`` for GNU compiler.

This property is initialized by the value of the
:variable:`CMAKE_PCH_WARN_INVALID` variable if it is set when a target is
created.  If that variable is not set, the property defaults to ``ON``.
