<LANG>_VISIBILITY_PRESET
------------------------

Value for symbol visibility compile flags

The ``<LANG>_VISIBILITY_PRESET`` property determines the value passed in a
visibility related compile option, such as ``-fvisibility=`` for ``<LANG>``.
This property affects compilation in sources of all types of targets.
See policy :policy:`CMP0063`.

This property is initialized by the value of the
:variable:`CMAKE_<LANG>_VISIBILITY_PRESET` variable if it is set when a
target is created.
