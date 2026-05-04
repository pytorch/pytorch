"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

Upstream SHA: 6bceaad2dba3b979b898824b146b1bb2816fc483 (quack 0.4.0)

Only the modules required by torch._native (rmsnorm and the gemm family)
are vendored. softmax, cross-entropy, rotary, topk, linear, etc. are
deliberately excluded. Imports within the vendored tree are rewritten to
be package-relative so this copy is independent of any ``quack``
top-level package that may be installed via pip, and ``torch.library``
op registrations are stripped so the vendored copy does not claim the
``quack::`` namespace at import time.

Gemm entry points live in ``torch._vendor.quack.gemm_interface`` — they
are not re-exported here to keep this module's import footprint small.
"""
__version__ = "0.4.0"

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see cute_dsl_mlir_threading).
from . import cute_dsl_elf_fix
from . import cute_dsl_mlir_threading

cute_dsl_elf_fix.patch()
cute_dsl_mlir_threading.patch()

from .rmsnorm import rmsnorm  # noqa: E402


__all__ = [
    "rmsnorm",
]
