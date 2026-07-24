# Generic registration of the pointwise definition table as aten CUDA overrides. Each row -> a
# (cond, impl) pair; the impl picks compute/output dtypes via aten's elementwise type
# promotion, bakes the row's scalar args as compute-dtype constants, and runs the one
# generic elementwise kernel. No per-op code beyond the table row.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._prims_common import elementwise_dtypes

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from .table import POINTWISE_DEF_TABLE, PointwiseDef


# The launch glue (_L), kernel runner (K), and op-math module (ops) all import
# `cutlass`, which `import torch` must not do (the lazy-DSL-import contract; see
# test_no_dsl_imports_after_import_torch). None is touched by a `cond` or by
# registration -- only by the `*_impl` closures on a real (non-declined) call, where
# the DSL runtime is present. table.py (imported above) is cutlass-free and carries
# the registration metadata; a row's `fn` is a NAME resolved via ops.get_fn() at call
# time. Bind the cutlass-laden modules as lazy proxies so their imports fire then.
if TYPE_CHECKING:
    from .._cutedsl import launch as _L
    from . import kernel as K, ops
else:
    _L = LazyModule("torch._native.ops._cutedsl.launch")
    K = LazyModule("torch._native.ops.pointwise.kernel")
    ops = LazyModule("torch._native.ops.pointwise.ops")


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _supported(t, dtypes) -> bool:
    # COW is not gated: inputs export read-only (launch._ro / ReadOnlyTensorWrapper),
    # so a COW input flows through the kernel without materializing.
    return (
        isinstance(t, torch.Tensor)
        and t.dtype in dtypes
        and t.is_contiguous()
        and t.numel() > 0
        and not cap.is_traced(t)
    )


def _scalars(row: PointwiseDef, args, kwargs):
    # Resolve the row's named scalar args, which the caller may pass either
    # positionally (after the nin tensors) or by keyword (e.g. torch.add(x, y,
    # alpha=2)). Missing -> the aten default (1 for add's alpha; the only scalar
    # default in the current table). Returns a tuple in row.scalars order.
    pos = args[row.nin :]
    out = []
    for i, name in enumerate(row.scalars):
        if name in kwargs:
            out.append(kwargs[name])
        elif i < len(pos):
            out.append(pos[i])
        else:
            out.append(1)
    return tuple(out)


def _make_cond(row: PointwiseDef):
    # The first `nin` args are always the positional tensor operands; serve only
    # when all are supported (row.dtypes or the family default), contiguous, on the
    # current device, and same-shape (no broadcast yet). Scalar args do not gate.
    dtypes = row.dtypes or _SUPPORTED

    def cond(*args, **kwargs):
        ins = args[: row.nin]
        if len(ins) < row.nin or not all(_supported(t, dtypes) for t in ins):
            return False
        if not (cap.device_ok(ins[0]) and cap.on_current_device(ins[0])):
            return False
        # A complex scalar arg (e.g. add's alpha) on real inputs is invalid for our
        # real-dtype kernels -- decline so aten raises its "argument alpha must not be
        # a complex number" error rather than us choking on the const bake.
        if any(isinstance(s, complex) for s in _scalars(row, args, kwargs)):
            return False
        # Every operand must share ins[0]'s shape (no broadcast yet) AND its device:
        # a CPU / other-device operand in a later slot would otherwise launch into the
        # CUDA kernel (illegal access). Declining lets aten raise its cross-device
        # error (or coerce a 0-d CPU scalar) instead.
        return all(
            t.shape == ins[0].shape and t.device == ins[0].device for t in ins[1:]
        )

    return cond


def _make_impl(row: PointwiseDef):
    # Promotion (compute / output dtypes) is a pure function of the input dtypes and
    # the row's fixed promotion kind -- NOT the shapes -- so memoize it per input-
    # dtype tuple. This keeps elementwise_dtypes (a non-trivial Python helper, ~4us)
    # off the hot path; the only per-call promotion work is then a dict lookup.
    promo_cache: dict = {}

    def _promo(in_dtypes):
        got = promo_cache.get(in_dtypes)
        if got is None:
            # elementwise_dtypes takes tensors/numbers; 0-d tensors of the dtype are
            # representative for the (tensor-only) promotion of our overrides.
            probes = [torch.empty(0, dtype=d, device="cuda") for d in in_dtypes]
            compute, out_dtype = elementwise_dtypes(
                *probes, type_promotion_kind=row.promotion
            )
            ct = _L.torch2cute[compute]
            out_dtypes = (
                row.out_dtypes(out_dtype) if row.out_dtypes else [out_dtype] * row.nout
            )
            got = (ct, compute, tuple(out_dtypes))
            promo_cache[in_dtypes] = got
        return got

    def impl(*args, **kwargs):
        ins = list(args[: row.nin])
        scalars = _scalars(row, args, kwargs)
        in_dtypes = tuple(t.dtype for t in ins)
        ct, compute, out_dtypes = _promo(in_dtypes)
        consts = tuple(ct(s) for s in scalars)
        # The operand LAYOUTS (per-operand shape + stride) are baked into the
        # compiled kernel, so they must be in the cache key -- distinct shapes /
        # broadcast patterns compile distinct kernels.
        # Alignment is in the key because it picks the PATH (vec wraps
        # assert 16B alignment and raise on sliced views); two calls
        # with identical shapes/strides but different storage offsets
        # must not share a plan. storage_offset, not data_ptr(): the
        # latter materializes COW inputs.
        key = (
            row.aten,
            in_dtypes,
            tuple((t.shape, t.stride()) for t in ins),
            tuple((t.storage_offset() * t.element_size()) % 16 == 0 for t in ins),
            scalars,
        )
        outs = K.run(
            ops.get_fn(row.fn),
            key,
            row.nin,
            row.nout,
            consts,
            ct,
            compute,
            ins,
            out_dtypes,
        )
        return outs[0] if row.nout == 1 else outs

    return impl


def register_pointwise_overrides() -> None:
    for row in POINTWISE_DEF_TABLE:
        cu.register_op_override(
            "aten", row.aten, "CUDA", cond=_make_cond(row), impl=_make_impl(row)
        )
