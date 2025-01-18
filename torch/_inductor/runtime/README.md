# `torch._inductor.runtime`

This folder contains code needed at runtime by the output code of
Inductor.  The output code of Inductor will import `torch` and
`torch._inductor.runtime`, but should not import from other files in
`torch._inductor.*`.  Note that this code includes code that is
needed to actually perform Triton compilation, but is not needed
in the actual, final runtime execution of kernels.

Runtime includes Triton/C++ generated code, which are compiled (sometimes in
parallel) when the output code of Inductor is imported.  It also includes
the autotuning code and heuristics to decide block sizes of generated code.

One of the original motivations for this directory split was so that the Triton
compile subprocesses could access Triton and our compiler support code while
mocking out most of `torch`, which can take seconds to import (sometimes more
than a Triton compile itself).  An abandoned prototype of this can be found
[here](https://github.com/pytorch/pytorch/pull/124682/files).
