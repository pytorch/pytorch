LINKER_LANGUAGE
---------------

Specifies language whose compiler will invoke the linker.

For executables, shared libraries, and modules, this sets the language
whose compiler is used to link the target (such as "C" or "CXX").  A
typical value for an executable is the language of the source file
providing the program entry point (main).  If not set, the language
with the highest linker preference value is the default.  Details of
the linker preferences are considered internal, but some limited
discussion can be found under the internal
:variable:`CMAKE_<LANG>_LINKER_PREFERENCE` variables.

If this property is not set by the user, it will be calculated at
generate-time by CMake.
