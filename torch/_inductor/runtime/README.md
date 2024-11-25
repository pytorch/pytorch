# `torch._inductor.runtime`

There's not really a coherent concept for this folder; in general it contains
code for:

1. Out-of-process Triton compilation (e.g., when doing multiprocess parallel
   Triton compile)

2. Code that is accessed by the code we generated in Inductor codegen (the
   "runtime", although not really a true runtime because we don't actually run
   this code when running, it's what the Triton compilation uses).

It isn't really a true runtime (aka, code that needs to be invoked when
actually running Inductor compiled kernels).

One of the original motivations for this directory split was so that the
subprocesses could access Triton and our compiler support code without having
to import torch.  However, we never actually go this to work, so there are
still plenty of torch imports in this folder.
