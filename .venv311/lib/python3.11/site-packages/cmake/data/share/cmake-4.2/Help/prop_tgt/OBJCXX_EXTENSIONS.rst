OBJCXX_EXTENSIONS
-----------------

.. versionadded:: 3.16

Boolean specifying whether compiler specific extensions are requested.

This property specifies whether compiler specific extensions should be
used.  For some compilers, this results in adding a flag such
as ``-std=gnu++11`` instead of ``-std=c++11`` to the compile line.  This
property is ``ON`` by default. The basic ObjC++ standard level is
controlled by the :prop_tgt:`OBJCXX_STANDARD` target property.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

If the property is not set, and the project has set the :prop_tgt:`CXX_EXTENSIONS`,
the value of :prop_tgt:`CXX_EXTENSIONS` is set for ``OBJCXX_EXTENSIONS``.

This property is initialized by the value of
the :variable:`CMAKE_OBJCXX_EXTENSIONS` variable if set when a target is
created and otherwise by the value of
:variable:`CMAKE_OBJCXX_EXTENSIONS_DEFAULT <CMAKE_<LANG>_EXTENSIONS_DEFAULT>`
(see :policy:`CMP0128`).
