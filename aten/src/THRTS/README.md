# THRTS: TorcH RunTime System

This library is in flux, but intuitively, its purpose in life is to contain
the "runtime system" of Torch: the bits that you need to actually run the
kernels in Torch, but without any of the kernels.

At the moment, this is:

- Some utilities for atomics (these should possibly be subsumed by the
  direct atomics APIs available in new standards of C and C++), and

- An allocator.

Should you put code in THRTS?  You should only do it if:

- It adds no dependencies, and

- It's not "operator" code, e.g., there should only be a constant amount
  of code, not a size that linearly scales with the number of operators
  in TH.

NB: The dependency set of this module today is the decidedly unminimal
OpenMP, BLAS and LAPACK.  It is probably possible to move some of these out.
