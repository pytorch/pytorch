CMAKE_<LANG>_LINKER_PREFERENCE
------------------------------

An internal variable subject to change.

Preference value for linker language selection.

The "linker language" for executable, shared library, and module
targets is the language whose compiler will invoke the linker.  The
:prop_tgt:`LINKER_LANGUAGE` target property sets the language explicitly.
Otherwise, the linker language is that whose linker preference value
is highest among languages compiled and linked into the target.  See
also the :variable:`CMAKE_<LANG>_LINKER_PREFERENCE_PROPAGATES` variable.
