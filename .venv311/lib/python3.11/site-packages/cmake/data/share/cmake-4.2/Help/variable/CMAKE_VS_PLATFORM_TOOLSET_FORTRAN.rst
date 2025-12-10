CMAKE_VS_PLATFORM_TOOLSET_FORTRAN
---------------------------------

.. versionadded:: 3.29

Fortran compiler to be used by Visual Studio projects.

:ref:`Visual Studio Generators` support selecting among Fortran compilers
that have the required Visual Studio Integration feature installed.  The
compiler may be specified by a field in :variable:`CMAKE_GENERATOR_TOOLSET` of
the form ``fortran=...``. CMake provides the selected Fortran compiler in this
variable.

If the field was not specified, the default depends on the generator:

* On :generator:`Visual Studio 18 2026` and above, the default is ``ifx``.

* On older :ref:`Visual Studio Generators`, the default is empty, which the
  Intel Visual Studio Integration interprets as equivalent to ``ifort``.
