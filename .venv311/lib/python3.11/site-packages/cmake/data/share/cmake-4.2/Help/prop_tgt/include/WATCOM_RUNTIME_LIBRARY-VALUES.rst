``SingleThreaded``
  Compile without additional flags to use a single-threaded
  statically-linked runtime library.
``SingleThreadedDLL``
  Compile with ``-br`` or equivalent flag(s) to use a single-threaded
  dynamically-linked runtime library. This is not available for Linux
  targets.
``MultiThreaded``
  Compile with ``-bm`` or equivalent flag(s) to use a multi-threaded
  statically-linked runtime library.
``MultiThreadedDLL``
  Compile with ``-bm -br`` or equivalent flag(s) to use a multi-threaded
  dynamically-linked runtime library. This is not available for Linux
  targets.

The value is ignored on non-Watcom compilers but an unsupported value will
be rejected as an error when using a compiler targeting the Watcom ABI.

The value may also be the empty string (``""``) in which case no runtime
library selection flag will be added explicitly by CMake.
