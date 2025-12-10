SKIP_UNITY_BUILD_INCLUSION
--------------------------

.. versionadded:: 3.16

Setting this property to true ensures the source file will be skipped by
unity builds when its associated target has its :prop_tgt:`UNITY_BUILD`
property set to true.  The source file will instead be compiled on its own
in the same way as it would with unity builds disabled.

This property helps with "ODR (One definition rule)" problems where combining
a particular source file with others might lead to build errors or other
unintended side effects.

Note that sources which are scanned for C++ modules (see
:manual:`cmake-cxxmodules(7)`) are not eligible for unity build inclusion and
will automatically be excluded.
