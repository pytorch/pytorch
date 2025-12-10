Fortran_MODULE_DIRECTORY
------------------------

Specify output directory for Fortran modules provided by the target.

If the target contains Fortran source files that provide modules and
the compiler supports a module output directory this specifies the
directory in which the modules will be placed.  When this property is
not set the modules will be placed in the build directory
corresponding to the target's source directory.  If the variable
:variable:`CMAKE_Fortran_MODULE_DIRECTORY` is set when a target is created its
value is used to initialize this property.

When using one of the :ref:`Visual Studio Generators` with the Intel Fortran
plugin installed in Visual Studio, a subdirectory named after the
configuration will be appended to the path where modules are created.
For example, if ``Fortran_MODULE_DIRECTORY`` is set to ``C:/some/path``,
modules will end up in ``C:/some/path/Debug`` (or
``C:/some/path/Release`` etc.) when an Intel Fortran ``.vfproj`` file is
generated, and in ``C:/some/path`` when any other generator is used.

Note that some compilers will automatically search the module output
directory for modules USEd during compilation but others will not.  If
your sources USE modules their location must be specified by
:prop_tgt:`INCLUDE_DIRECTORIES` regardless of this property.
