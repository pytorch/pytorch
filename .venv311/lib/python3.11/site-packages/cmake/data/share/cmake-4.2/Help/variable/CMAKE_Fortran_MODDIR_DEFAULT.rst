CMAKE_Fortran_MODDIR_DEFAULT
----------------------------

Fortran default module output directory.

Most Fortran compilers write ``.mod`` files to the current working
directory.  For those that do not, this is set to ``.`` and used when
the :prop_tgt:`Fortran_MODULE_DIRECTORY` target property is not set.
