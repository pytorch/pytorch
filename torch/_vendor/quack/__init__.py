"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

The pinned upstream commit is recorded in ``__upstream_sha__`` below and is
sourced from ``PINNED_SHA`` in tools/vendoring/quack/vendor.sh. The
vendoring script verifies that commit is reachable from Dao-AILab/quack main
before applying the local FlexGEMM patchset. Only the modules required by
torch._native.ops.norm.rmsnorm_impl, torch._inductor.kernel.flex_gemm, and the
symmetric GEMM are vendored. Imports are rewritten to absolute
torch._vendor.quack imports so this copy is independent of any ``quack``
top-level package that may be installed via pip. Custom op namespaces are
renamed from ``quack::`` to ``torch_vendor_quack::`` for the same reason.
"""
__version__ = "0.6.4"
__upstream_sha__ = "4709411169dcc3dc4e23f8f32f385eb4b6871d9d"

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see cute_dsl_mlir_threading).
from . import cute_dsl_elf_fix
from . import cute_dsl_mlir_threading

cute_dsl_elf_fix.patch()
cute_dsl_mlir_threading.patch()

def __getattr__(name):
    if name == "rmsnorm":
        from .rmsnorm import rmsnorm

        return rmsnorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "rmsnorm",
]
