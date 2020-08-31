
# libtorchpy
Intended to be the user-facing library, which can be dynamically linked by an application, and should not bring visible baggage in the form of dynamically linking python or depending on globs of .py files.

This is the layer where multiple interpreters can be instantiated, and logic to round-robin, bind-to-thread, share state across interpreters, etc., can be orchestrated.


# libinterpreter
This library is mainly an implementation detail and not intended for direct use.  It contains a compelete copy of an interpreter, so that when multiple copies of libinterpreter are instantiated there are multiple interpreters.

# build system
Currently only the cmake build system is supported.

CPython is built in-source-tree as a cmake custom command.  CPython was added as a git submodule rather than downloaded as part of cmake, to be consistent with other submodules.  custom-command was preferred over custom-target as it allows specifying output files which makes cmake automatically configure dependencies.  CMake external project was not used as I couldn't figure out how to set up this dependency on .o files rather than final libs/execs.
