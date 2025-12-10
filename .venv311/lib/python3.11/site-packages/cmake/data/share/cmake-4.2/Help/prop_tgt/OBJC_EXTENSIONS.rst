OBJC_EXTENSIONS
---------------

.. versionadded:: 3.16

Boolean specifying whether compiler specific extensions are requested.

This property specifies whether compiler specific extensions should be
used.  For some compilers, this results in adding a flag such
as ``-std=gnu11`` instead of ``-std=c11`` to the compile line.  This
property is ``ON`` by default. The basic OBJC standard level is
controlled by the :prop_tgt:`OBJC_STANDARD` target property.

If the property is not set, and the project has set the :prop_tgt:`C_EXTENSIONS`,
the value of :prop_tgt:`C_EXTENSIONS` is set for ``OBJC_EXTENSIONS``.

See the :manual:`cmake-compile-features(7)` manual for information on
compile features and a list of supported compilers.

This property is initialized by the value of
the :variable:`CMAKE_OBJC_EXTENSIONS` variable if set when a target is created
and otherwise by the value of
:variable:`CMAKE_OBJC_EXTENSIONS_DEFAULT <CMAKE_<LANG>_EXTENSIONS_DEFAULT>`
(see :policy:`CMP0128`).
