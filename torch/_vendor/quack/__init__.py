"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

The pinned upstream commit is recorded in ``__upstream_sha__`` below and is
sourced from ``PINNED_SHA`` in tools/vendoring/quack/vendor.sh. Only the
modules required by torch._native.ops.norm.rmsnorm_impl and selected GEMM
epilogue implementation paths are vendored. Imports are rewritten to be package-relative
so this copy is independent of any ``quack`` top-level package that may be
installed via pip. Custom op namespaces are renamed from ``quack::`` to
``torch_vendor_quack::`` for the same reason.
"""
__version__ = "0.4.1"
__upstream_sha__ = "77e72af5565cd7aec2132944bb001de2c358617a"

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
