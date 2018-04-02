# THCRTS: TorcH CUDA RunTime System

This library is in flux, but intuitively, its purpose in life is to contain
the "runtime system" of THC: the bits that you need to actually run the
kernels in THC, but without any of the kernels.  This library has a CUDA
dependency, for obvious reasons.

At the moment, this is:

- A caching allocator.  This is the secret sauce of Torch!  You might find
  it useful.

Should you put code in THCRTS?  You should only do it if:

- It adds no dependencies, and

- It's not "operator" code, e.g., there should only be a constant amount
  of code, not a size that linearly scales with the number of operators
  in THC.
