``MultiThreaded``
  Compile with ``-MT`` or equivalent flag(s) to use a multi-threaded
  statically-linked runtime library.
``MultiThreadedDLL``
  Compile with ``-MD`` or equivalent flag(s) to use a multi-threaded
  dynamically-linked runtime library.
``MultiThreadedDebug``
  Compile with ``-MTd`` or equivalent flag(s) to use a multi-threaded
  statically-linked runtime library.
``MultiThreadedDebugDLL``
  Compile with ``-MDd`` or equivalent flag(s) to use a multi-threaded
  dynamically-linked runtime library.

The value is ignored on compilers not targeting the MSVC ABI, but an
unsupported value will be rejected as an error when using a compiler
targeting the MSVC ABI.

The value may also be the empty string (``""``) in which case no runtime
library selection flag will be added explicitly by CMake.  Note that with
:ref:`Visual Studio Generators` the native build system may choose to
add its own default runtime library selection flag.
