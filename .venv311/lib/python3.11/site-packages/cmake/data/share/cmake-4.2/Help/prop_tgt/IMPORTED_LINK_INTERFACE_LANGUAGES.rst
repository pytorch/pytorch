IMPORTED_LINK_INTERFACE_LANGUAGES
---------------------------------

Languages compiled into an ``IMPORTED`` static library.

Set this to the list of languages of source files compiled to produce
a ``STATIC IMPORTED`` library (such as ``C`` or ``CXX``).  CMake accounts for
these languages when computing how to link a target to the imported
library.  For example, when a C executable links to an imported C++
static library CMake chooses the C++ linker to satisfy language
runtime dependencies of the static library.

This property is ignored for targets that are not ``STATIC`` libraries.
This property is ignored for non-imported targets.
