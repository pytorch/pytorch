ASM<DIALECT>FLAGS
-----------------

.. include:: include/ENV_VAR.rst

Add default compilation flags to be used when compiling a specific dialect
of an assembly language.  ``ASM<DIALECT>FLAGS`` can be one of:

* ``ASMFLAGS``
* ``ASM_NASMFLAGS``
* ``ASM_MASMFLAGS``
* ``ASM_MARMASMFLAGS``
* ``ASM-ATTFLAGS``

.. |CMAKE_LANG_FLAGS| replace:: :variable:`CMAKE_ASM<DIALECT>_FLAGS <CMAKE_<LANG>_FLAGS>`
.. |LANG| replace:: ``ASM<DIALECT>``
.. include:: include/LANG_FLAGS.rst

See also :variable:`CMAKE_ASM<DIALECT>_FLAGS_INIT <CMAKE_<LANG>_FLAGS_INIT>`.
