"""AOT builder for the reduction family: K1 row-reduce, last-dim, final.

Phase 1 of AOT-ing the reductions (see PAIN_POINTS P5/P7): only the
one-shot row kernel is exported -- it is already dynamic-M with the
grid computed inside the @cute.jit region, so the exported launcher
needs no host planning. Reduce-all (xcta two-stage, scratch + C-split
planning in C++), column (K2), index traits, and the K0 general path
stay JIT.

The kernel body is the same RowReduce the JIT route compiles
(same-kernel by construction; two-stage build). ``_Adapter1`` exists
only because export_to_c cannot emit a C ABI for the kernel's
``mOuts: list`` argument (PAIN_POINTS P11).

Grid axes: trait (sum/mean/amax/amin/prod) x dtype {float32, bfloat16}
(user decision: fp16/fp64 stay JIT) x N buckets. N is a compile-time
constant of RowReduce (const_expr vec/tile selection), so off-bucket N
stays JIT -- the finite-grid compromise of P5.

The kernel accumulates and stores in fp32 (the _acc_policy the JIT
uses); for bf16 inputs the C++ launch allocates an fp32 scratch and
casts into aten's bf16 out, exactly the .to() the JIT impl pays.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass.cute as cute
from cutlass import Float32

from .._cutedsl import traits as T
from . import kernel_row


_DTYPES_CUTE = {}  # populated lazily; cutlass dtype objects


def _cute_dtype(name: str):
    if not _DTYPES_CUTE:
        import cutlass

        _DTYPES_CUTE.update({"float32": cutlass.Float32, "bfloat16": cutlass.BFloat16})
    return _DTYPES_CUTE[name]


_DTYPE_SHORT = {"float32": "f32", "bfloat16": "bf16"}

_TRAITS = {
    "sum": lambda: T.SumOps(acc=Float32),
    "mean": lambda: T.MeanOps(acc=Float32),
    "amax": lambda: T.AMaxOps(acc=Float32),
    "amin": lambda: T.AMinOps(acc=Float32),
    "prod": lambda: T.ProdOps(acc=Float32),
}


class _Adapter1:
    # export_to_c cannot marshal list arguments; forward a fixed-arity
    # signature into RowReduce's (mX, mOuts: list, stream).
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut0: cute.Tensor, stream: cuda.CUstream):
        self.op(mX, [mOut0], stream)


def build(spec: dict) -> dict:
    """One grid point -> compile inputs + marshalling sidecar.

    spec: {"trait": sum|mean|amax|amin|prod, "dtype": name, "N": int}.
    M (rows) is dynamic: mode 0 of the (M, N) input and the (M,) output
    are cute.sym_int, so one kernel per point serves every M.
    """
    trait = _TRAITS[spec["trait"]]()
    dtype = _cute_dtype(spec["dtype"])
    N = int(spec["N"])
    op = kernel_row.RowReduce(trait, dtype, N)
    adapter = _Adapter1(op)
    # Bucketed N is a multiple of the vector width, so rows are 16B
    # apart and the wide-load path is legal; the C++ prelude enforces
    # the matching base-pointer alignment.
    x_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(N, 1), assumed_align=16
    )
    # Kernel out = the fp32 accumulator dtype (see module docstring).
    out_fake = cute.runtime.make_fake_tensor(
        Float32, (cute.sym_int(),), stride=(1,), assumed_align=4
    )
    prefix = f"red_{spec['trait']}_{_DTYPE_SHORT[spec['dtype']]}_n{N}"
    return {
        "prefix": prefix,
        "fn": adapter,
        "fake_args": [x_fake, out_fake, cute.runtime.make_fake_stream()],
        "tensor_args": [
            {"name": "mX", "dynamic_sizes": [0], "read_only": True},
            {"name": "mOut0", "dynamic_sizes": [0]},
        ],
    }
