PCH_INSTANTIATE_TEMPLATES
-------------------------

.. versionadded:: 3.19

When this property is set to true, the precompiled header compiler options
will contain a flag to instantiate templates during the generation of the PCH
if supported. This can significantly improve compile times. Supported in Clang
since version 11.

This property is initialized by the value of the
:variable:`CMAKE_PCH_INSTANTIATE_TEMPLATES` variable if it is set when a target
is created.  If that variable is not set, the property defaults to ``ON``.
