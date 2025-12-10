CMAKE_<LANG>_LINKER_PREFERENCE_PROPAGATES
-----------------------------------------

An internal variable subject to change.

True if :variable:`CMAKE_<LANG>_LINKER_PREFERENCE` propagates across targets.

This is used when CMake selects a linker language for a target.
Languages compiled directly into the target are always considered.  A
language compiled into static libraries linked by the target is
considered if this variable is true.
