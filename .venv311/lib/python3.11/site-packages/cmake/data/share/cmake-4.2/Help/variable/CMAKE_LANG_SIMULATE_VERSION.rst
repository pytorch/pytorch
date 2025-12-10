CMAKE_<LANG>_SIMULATE_VERSION
-----------------------------

Version string of "simulated" compiler.

Some compilers simulate other compilers to serve as drop-in
replacements.  When CMake detects such a compiler it sets this
variable to what would have been the :variable:`CMAKE_<LANG>_COMPILER_VERSION`
for the simulated compiler.
