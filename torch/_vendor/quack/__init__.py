"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

Upstream SHA: 6bceaad2dba3b979b898824b146b1bb2816fc483 (quack 0.4.0)

Only the modules required by torch._native.ops.norm.rmsnorm_impl are vendored.
Imports are rewritten to be package-relative so this copy is independent of any
``quack`` top-level package that may be installed via pip. Custom op namespaces
are renamed from ``quack::`` to ``torch_vendor_quack::`` for the same reason.
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
