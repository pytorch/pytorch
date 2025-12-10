.. note::
 This property does not apply to STATIC library targets because no linker
 is invoked to produce them so they have no linker-generated ``.pdb`` file
 containing debug symbols.

 The linker-generated program database files are specified by the
 ``/pdb`` linker flag and are not the same as compiler-generated
 program database files specified by the ``/Fd`` compiler flag.
 Use the |COMPILE_PDB_XXX| property to specify the latter.
