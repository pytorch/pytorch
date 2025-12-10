Fortran_FORMAT
--------------

Set to ``FIXED`` or ``FREE`` to indicate the Fortran source layout.

This property tells CMake whether the Fortran source files in a target
use fixed-format or free-format.  CMake will pass the corresponding
format flag to the compiler.  Use the source-specific ``Fortran_FORMAT``
property to change the format of a specific source file.  If the
variable :variable:`CMAKE_Fortran_FORMAT` is set when a target is created its
value is used to initialize this property.
