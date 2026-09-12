"""One-time CuTeDSL JIT-engine warmup, so export_to_c works for any arch.

export_to_c needs LLVM state that the DSL only sets up when it builds a
JIT engine, and base_dsl/dsl.py builds one only when `num_kernels == 0 or
compile_gpu_arch == envar.arch`. So a cross-arch export as the first
compile in a process fails with "Failed to dump object file with PIC
relocation", while the same call succeeds after any engine-creating
compile. A kernel-free jit function hits the num_kernels == 0 branch:
~0.12s, once per process, no CUDA device needed.

Its own module because toolchains.py has `from __future__ import
annotations`, which stringifies the annotation below and breaks the DSL's
signature introspection.
"""

import cutlass.cute as cute
from cutlass import Float32


@cute.jit
def _no_kernel(a: Float32):
    # No .launch() anywhere, so num_kernels stays 0.
    return a + Float32(1.0)


def warm_up() -> None:
    cute.compile(_no_kernel, Float32(1.0))
