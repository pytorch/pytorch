# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch

from torch._higher_order_ops.hints_wrap import hints_wrapper


_SUPPORTED_GEMM_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_mm.default,
}


def gemm_epilogue_fusion(
    gemm_op: torch._ops.OpOverload,
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
):
    if gemm_op not in _SUPPORTED_GEMM_OPS:
        raise RuntimeError(f"unsupported GEMM op for epilogue fusion: {gemm_op}")

    if gemm_kwargs is None:
        gemm_kwargs = {}

    def body_fn(*args):
        return epilogue_fn(gemm_op(*args, **gemm_kwargs))

    return hints_wrapper(
        body_fn,
        gemm_args,
        {},
        hints={"gemm_epilogue_fusion": True, "must_fuse": True},
    )
