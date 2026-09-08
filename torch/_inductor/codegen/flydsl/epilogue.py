# mypy: allow-untyped-defs
"""Materialize accumulator-only Inductor epilogues as FlyDSL expressions."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import OpsHandler
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    import torch
    from torch._inductor.scheduler import BaseSchedulerNode


class _EpilogueOps(OpsHandler[str]):
    def __init__(self, accumulator_name: str):
        self.accumulator_name = accumulator_name

    def _binary(self, op: str, x0: str, x1: str) -> str:
        return f"({x0} {op} {x1})"

    def load(self, name: str, index: str) -> str:
        if name != self.accumulator_name:
            raise NotImplementedError(
                "FlyDSL GEMM epilogues only support accumulator reads"
            )
        return "acc"

    def constant(self, value: bool | float | int, dtype: torch.dtype) -> str:
        if isinstance(value, float) and not math.isfinite(value):
            raise NotImplementedError(
                "FlyDSL GEMM epilogues require finite scalar constants"
            )
        return repr(value)

    def add(self, x0: str, x1: str) -> str:
        return self._binary("+", x0, x1)

    def sub(self, x0: str, x1: str) -> str:
        return self._binary("-", x0, x1)

    def mul(self, x0: str, x1: str) -> str:
        return self._binary("*", x0, x1)

    def neg(self, x0: str) -> str:
        return f"-({x0})"

    def gt(self, x0: str, x1: str) -> str:
        return self._binary(">", x0, x1)

    def where(self, condition: str, input: str, other: str) -> str:
        return f"({condition}).select({input}, {other})"

    def maximum(self, x0: str, x1: str) -> str:
        return self.where(self.gt(x0, x1), x0, x1)

    def relu(self, x0: str) -> str:
        return self.maximum(x0, "0")


def materialize_flydsl_scheduler_epilogue(
    original_buffer_name: str,
    epilogue_nodes: list[BaseSchedulerNode],
) -> str:
    """Render supported scheduler nodes as a capture-free constexpr lambda."""
    if len(epilogue_nodes) != 1:
        raise NotImplementedError("FlyDSL GEMM supports one epilogue node")

    scheduler_nodes = list(epilogue_nodes[0].get_nodes())
    if len(scheduler_nodes) != 1:
        raise NotImplementedError("FlyDSL GEMM supports one epilogue node")
    ir_node = scheduler_nodes[0].node
    if not isinstance(ir_node, ComputedBuffer) or not isinstance(
        ir_node.data, Pointwise
    ):
        raise NotImplementedError("FlyDSL GEMM epilogue must be pointwise")

    handler = _EpilogueOps(original_buffer_name)
    with V.set_ops_handler(handler):
        result = ir_node.data.inner_fn(*ir_node.data.inner_fn_args())
    result = str(result)
    key = hashlib.sha256(result.encode()).hexdigest()
    return (
        "HAS_EPILOGUE: fx.Constexpr = True\n"
        f"EPILOGUE_KEY: fx.Constexpr = {key!r}\n"
        f"EPILOGUE_FN = lambda acc: {result}\n"
    )
