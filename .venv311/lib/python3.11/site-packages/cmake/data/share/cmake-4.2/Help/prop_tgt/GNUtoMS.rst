GNUtoMS
-------

Convert GNU import library (``.dll.a``) to MS format (``.lib``).

When linking a shared library or executable that exports symbols using
GNU tools on Windows (MinGW/MSYS) with Visual Studio installed convert
the import library (``.dll.a``) from GNU to MS format (``.lib``).  Both import
libraries will be installed by :command:`install(TARGETS)` and exported by
:command:`install(EXPORT)` and  :command:`export` to be linked
by applications with either GNU- or MS-compatible tools.

If the variable ``CMAKE_GNUtoMS`` is set when a target is created its
value is used to initialize this property.  The variable must be set
prior to the first command that enables a language such as :command:`project`
or :command:`enable_language`.  CMake provides the variable as an option to the
user automatically when configuring on Windows with GNU tools.
