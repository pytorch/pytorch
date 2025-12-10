``CMAKE_REQUIRED_FLAGS``
  A space-separated string of additional flags to pass to the compiler.
  A :ref:`semicolon-separated list <CMake Language Lists>` will not work.
  The contents of :variable:`CMAKE_<LANG>_FLAGS` and its associated
  configuration-specific :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>` variables
  are automatically prepended to the compiler command before the contents of
  this variable.
