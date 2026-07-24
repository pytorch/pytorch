"""AOT builder for the pointwise family: fixed-arity export adapters
over the generic vec kernel.

Deltas from the JIT route, forced by the export surface:

* ``_ElementwiseVec`` takes ``mIns: list, mOuts: list`` -- fine for
  cute.compile, but ``export_to_c`` cannot emit a C ABI for list
  arguments (PAIN_POINTS P11). ``_Adapter<nin>to<nout>`` wraps the op
  in an explicit-arity @cute.jit signature forwarding as lists; the
  kernel body underneath is the same one the JIT route compiles.
* The JIT route bakes each operand's layout per compile (plan cache
  keyed on shapes/strides). The AOT compile marks mode 0 of the
  (nvec, V) view dynamic so ONE kernel per (op, dtype tuple) serves
  every size on the vec path (PAIN_POINTS P1); numel % V == 0 and
  per-operand alignment are coverage conditions.

Dtype axis: the grid enumerates INPUT-DTYPE TUPLES over {float32,
bfloat16} (mixed tuples included). Promotion is resolved HERE at
export time via the same ``elementwise_dtypes`` the JIT impl calls --
compute dtype and out dtype are baked per point, so the generated C++
matches dtype tuples by equality and never restates promotion
(PAIN_POINTS P8, corrected). On this grid: compute is always fp32; out
is bf16 iff every input is bf16, else fp32.

The vector width V is shared across operands (one flat (numel/V, V)
view per tensor), so V = 128 bits / WIDEST element width in the tuple
(fp32 present -> V=4; all-bf16 -> V=8). Narrow (bf16) operands in a
mixed tuple then load 64 bits per lane -- correct, slightly sub-peak.
Each operand's assumed_align follows its own V * elem_bytes.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass.cute as cute
from cutlass import BFloat16, Float32

import torch
from torch._prims_common import elementwise_dtypes

from .kernel import _ElementwiseVec
from .ops import get_fn
from .table import POINTWISE_DEF_TABLE


_CUTE = {"float32": Float32, "bfloat16": BFloat16}
_TORCH = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_TORCH_TO_NAME = {v: k for k, v in _TORCH.items()}
_DTYPE_SHORT = {"float32": "f32", "bfloat16": "bf16"}


class _Adapter1to1:
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(self, mIn0: cute.Tensor, mOut0: cute.Tensor, stream: cuda.CUstream):
        self.op([mIn0], [mOut0], stream)


class _Adapter2to1:
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(
        self,
        mIn0: cute.Tensor,
        mIn1: cute.Tensor,
        mOut0: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.op([mIn0, mIn1], [mOut0], stream)


class _Adapter3to1:
    def __init__(self, op):
        self.op = op

    @cute.jit
    def __call__(
        self,
        mIn0: cute.Tensor,
        mIn1: cute.Tensor,
        mIn2: cute.Tensor,
        mOut0: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.op([mIn0, mIn1, mIn2], [mOut0], stream)


_ADAPTERS = {1: _Adapter1to1, 2: _Adapter2to1, 3: _Adapter3to1}


def _row(aten: str):
    for r in POINTWISE_DEF_TABLE:
        if r.aten == aten:
            return r
    raise KeyError(f"no pointwise table row for {aten!r}")


def promote(row_aten: str, in_names: tuple) -> tuple:
    """(compute torch dtype, out torch dtype) for an input-dtype tuple,
    via the SAME elementwise_dtypes call the JIT impl uses. Shared with
    aot.py's coverage/codegen so all three consumers agree."""
    row = _row(row_aten)
    probes = [torch.empty(0, dtype=_TORCH[n], device="meta") for n in in_names]
    return elementwise_dtypes(*probes, type_promotion_kind=row.promotion)


def vec_width(in_names: tuple) -> int:
    """Shared V across operands: 128 bits / widest element width."""
    widest = max(_TORCH[n].itemsize for n in in_names)
    return 16 // widest


def _fake_vec(dtype, V: int):
    # (nvec dynamic, V static); alignment = one V-wide row of THIS dtype.
    return cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), V), stride=(V, 1), assumed_align=V * dtype.width // 8
    )


def build(spec: dict) -> dict:
    """One grid point -> compile inputs + marshalling sidecar.

    spec: {"aten": table-row op symbol, "in_dtypes": tuple of dtype
    names, one per tensor input}. Scalar args (add's alpha etc.) are
    baked at their aten DEFAULT (1); non-default scalars stay JIT.
    """
    row = _row(spec["aten"])
    in_names = tuple(spec["in_dtypes"])
    assert len(in_names) == row.nin  # noqa: S101
    compute_t, out_t = promote(spec["aten"], in_names)
    compute = _CUTE[_TORCH_TO_NAME[compute_t]]
    out_cute = _CUTE[_TORCH_TO_NAME[out_t]]
    V = vec_width(in_names)
    consts = tuple(compute(1) for _ in row.scalars)
    op = _ElementwiseVec(
        get_fn(row.fn), row.nin, row.nout, consts, compute, [out_cute] * row.nout, V
    )
    adapter = _ADAPTERS[row.nin](op)
    in_fakes = [_fake_vec(_CUTE[n], V) for n in in_names]
    out_fake = _fake_vec(out_cute, V)
    decl_id = row.aten.replace(".", "_")
    tag = "_".join(_DTYPE_SHORT[n] for n in in_names)
    prefix = f"pw_{decl_id}_{tag}"
    return {
        "prefix": prefix,
        "fn": adapter,
        "fake_args": [*in_fakes, out_fake, cute.runtime.make_fake_stream()],
        "tensor_args": [
            {"name": f"mIn{i}", "dynamic_sizes": [0], "read_only": True}
            for i in range(row.nin)
        ]
        + [{"name": "mOut0", "dynamic_sizes": [0]}],
    }
