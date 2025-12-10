``Embedded``
  Compile with ``-Z7`` or equivalent flag(s) to produce object files
  with full symbolic debugging information.
``ProgramDatabase``
  Compile with ``-Zi`` or equivalent flag(s) to produce a program
  database that contains all the symbolic debugging information.
``EditAndContinue``
  Compile with ``-ZI`` or equivalent flag(s) to produce a program
  database that supports the Edit and Continue feature.

The value is ignored on compilers not targeting the MSVC ABI, but an
unsupported value will be rejected as an error when using a compiler
targeting the MSVC ABI.

The value may also be the empty string (``""``), in which case no debug
information format flag will be added explicitly by CMake.
