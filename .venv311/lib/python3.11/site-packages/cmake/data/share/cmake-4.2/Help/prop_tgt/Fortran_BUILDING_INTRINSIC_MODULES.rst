Fortran_BUILDING_INTRINSIC_MODULES
----------------------------------

.. versionadded:: 4.0

Instructs the CMake Fortran preprocessor that the target is building
Fortran intrinsics for building a Fortran compiler.

This property is off by default and should be turned only on projects
that build a Fortran compiler. It should not be turned on for projects
that use a Fortran compiler.

Turning this property on will correctly add dependencies for building
Fortran intrinsic modules whereas turning the property off will ignore
Fortran intrinsic modules in the dependency graph as they are supplied
by the compiler itself.
