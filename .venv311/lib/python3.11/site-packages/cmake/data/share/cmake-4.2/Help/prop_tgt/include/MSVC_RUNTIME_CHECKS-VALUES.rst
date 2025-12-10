``PossibleDataLoss``
  Compile with ``-RTCc`` or equivalent flag(s) to enable possible
  data loss checks.
``StackFrameErrorCheck``
  Compile with ``-RTCs`` or equivalent flag(s) to enable stack frame
  error checks.
``UninitializedVariable``
  Compile with ``-RTCu`` or equivalent flag(s) to enable uninitialized
  variables checks.

The value is ignored on compilers not targeting the MSVC ABI, but an
unsupported value will be rejected as an error when using a compiler
targeting the MSVC ABI.

The value may also be the empty string (``""``), in which case no runtime
error check flags will be added explicitly by CMake.
