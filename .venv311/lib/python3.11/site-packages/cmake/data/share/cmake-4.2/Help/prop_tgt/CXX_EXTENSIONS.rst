CXX_EXTENSIONS
--------------

.. versionadded:: 3.1

Boolean specifying whether compiler specific extensions are requested.

This property specifies whether compiler specific extensions should be
used.  For some compilers, this results in adding a flag such
as ``-std=gnu++11`` instead of ``-std=c++11`` to the compile line.  This
property is ``ON`` by default. The basic C++ standard level is
controlled by the :prop_tgt:`CXX_STANDARD` target property.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_CXX_EXTENSIONS` variable if set when a target is created
and otherwise by the value of
:variable:`CMAKE_CXX_EXTENSIONS_DEFAULT <CMAKE_<LANG>_EXTENSIONS_DEFAULT>` (see
:policy:`CMP0128`).
