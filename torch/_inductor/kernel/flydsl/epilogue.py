# mypy: allow-untyped-defs
from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import OpsHandler
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    import torch
    from torch._inductor.scheduler import BaseSchedulerNode


class _Expr:
    def __init__(self, source: str):
        self.source = source

    def __str__(self) -> str:
        return self.source


class _EpilogueOps(OpsHandler[Any]):
    def __init__(self, accumulator_name: str):
        self.accumulator_name = accumulator_name

    def _binary(self, op: str, a: Any, b: Any) -> _Expr:
        return _Expr(f"({a} {op} {b})")

    def _unary(self, op: str, x: Any) -> _Expr:
        return _Expr(f"{op}({x})")

    def load(self, name: str, index: Any) -> _Expr:
        if name != self.accumulator_name:
            raise NotImplementedError(
                "FlyDSL GEMM epilogues only support accumulator reads"
            )
        return _Expr("acc")

    def constant(self, value: Any, dtype: torch.dtype) -> Any:
        return repr(value)

    def add(self, a: Any, b: Any, *, alpha: Any = 1) -> _Expr:
        if alpha != 1:
            b = self.mul(b, alpha)
        return self._binary("+", a, b)

    def sub(self, a: Any, b: Any, *, alpha: Any = 1) -> _Expr:
        if alpha != 1:
            b = self.mul(b, alpha)
        return self._binary("-", a, b)

    def mul(self, a: Any, b: Any) -> _Expr:
        return self._binary("*", a, b)

    def truediv(self, a: Any, b: Any) -> _Expr:
        return self._binary("/", a, b)

    def neg(self, x: Any) -> _Expr:
        return self._unary("-", x)


def materialize_flydsl_scheduler_epilogue(
    original_buffer_name: str,
    epilogue_nodes: list[BaseSchedulerNode],
) -> tuple[str, str]:
    if not epilogue_nodes:
        return "", (
            "HAS_EPILOGUE: fx.Constexpr = False\n"
            "EPILOGUE_KEY: fx.Constexpr = 'identity'\n"
            "EPILOGUE_FN = lambda acc: acc\n"
        )

    env: dict[str, Any] = {original_buffer_name: _Expr("acc")}
    handler = _EpilogueOps(original_buffer_name)
    for scheduler_group in epilogue_nodes:
        if scheduler_group.is_reduction():
            raise NotImplementedError("FlyDSL GEMM epilogue reductions unsupported")
        for scheduler_node in scheduler_group.get_nodes():
            ir_node = scheduler_node.node
            if not isinstance(ir_node, ComputedBuffer) or not isinstance(
                ir_node.data, Pointwise
            ):
                raise NotImplementedError("FlyDSL GEMM epilogue must be pointwise")
            with V.set_ops_handler(handler):
                result = ir_node.data.inner_fn(*ir_node.data.inner_fn_args())
            env[ir_node.get_name()] = result
            env[scheduler_group.get_name()] = result

    final_name = epilogue_nodes[-1].get_name()
    if final_name not in env:
        raise AssertionError(f"missing final FlyDSL epilogue value {final_name}")

    result = str(env[final_name])
    key = hashlib.sha256(result.encode()).hexdigest()[:16]
    return (
        key,
        "HAS_EPILOGUE: fx.Constexpr = True\n"
        f"EPILOGUE_KEY: fx.Constexpr = {key!r}\n"
        f"EPILOGUE_FN = lambda acc: {result}\n",
    )
