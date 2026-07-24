# The pointwise op definition table (POINTWISE_DEF_TABLE): one PointwiseDef row per
# aten elementwise op. Each row is fully declarative -- the generic registration
# machinery (overrides.py) turns it into a (cond, impl) override, so adding an op is
# one row plus its kernel function in ops.py, not a hand-written override.
#
# This module is deliberately cutlass-FREE: it holds only the metadata registration
# needs (aten name, arity, promotion kind, scalar args, output-dtype policy) so that
# `import torch` -> override registration can read the table without pulling in the
# DSL runtime (the lazy-DSL-import contract; see test_no_dsl_imports_after_import_torch).
# The actual kernel math lives in ops.py, where every `fn` is a @cute.jit function;
# a row references it BY NAME (`fn`, a str) and overrides.py resolves the callable
# lazily via ops.get_fn(name) on the first real (non-declined) call.
#
# The named ops.py function is @cute.jit-able over COMPUTE-dtype scalars:
#   fn(*input_vals, *scalar_consts) -> result | tuple-of-results
# Inputs arrive already converted to the compute dtype; baked scalar args (e.g. add's
# `alpha`) follow, as compute-dtype constants. The result is cast to the op's output
# dtype. `fn` references only DSL ops (operators, cute.math.*), never a user-class
# method (which would trip the IR flattener).

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind


if TYPE_CHECKING:
    from collections.abc import Callable


class PointwiseDef(NamedTuple):
    aten: str  # aten op symbol incl overload, e.g. "add.Tensor", "neg"
    nin: int  # number of tensor inputs
    fn: str  # name of the @cute.jit kernel function in ops.py (resolved lazily)
    # aten elementwise type-promotion kind (single value, not combinable: a closed
    # Enum keying torch's elementwise_dtypes algorithm, not a bitwise Flag).
    promotion: PromotionKind = PromotionKind.DEFAULT
    scalars: tuple[str, ...] = ()  # positional arg names baked as compute consts
    nout: int = 1  # number of outputs (>1: e.g. frexp)
    # ESCAPE HATCH for ops whose output dtypes are NOT all the promotion result
    # (e.g. frexp -> (float mantissa, int32 exponent)). Maps the promotion result
    # torch dtype -> list[torch dtype] of length nout. None -> every output uses the
    # promotion result dtype (the common case).
    out_dtypes: Callable | None = None
    # Restrict the INPUT dtypes this override serves; inputs outside the set fall
    # back to aten. None -> the family default (all supported floats). Use to narrow
    # an op whose kernel is only correct for some dtypes (e.g. frexp excludes fp64).
    dtypes: tuple | None = None


# Row data lives in _table_data.py as torch-free tuples (the AOT
# declaration factory reads it at torchgen time, where torch is not
# importable -- see PAIN_POINTS P13); this module rebuilds the typed
# rows with the real PromotionKind and callable escape hatches.
from ._table_data import PW_ROWS


_OUT_DTYPES = {"frexp": lambda compute: [compute, torch.int32]}
_DTYPE_SETS = {"no_fp64": (torch.float16, torch.bfloat16, torch.float32)}

POINTWISE_DEF_TABLE: tuple[PointwiseDef, ...] = tuple(
    PointwiseDef(
        aten,
        nin,
        fn,
        promotion=PromotionKind[promotion],
        scalars=scalars,
        nout=nout,
        out_dtypes=_OUT_DTYPES.get(out_tag),
        dtypes=_DTYPE_SETS.get(dt_tag),
    )
    for (aten, nin, fn, promotion, scalars, nout, out_tag, dt_tag) in PW_ROWS
)
