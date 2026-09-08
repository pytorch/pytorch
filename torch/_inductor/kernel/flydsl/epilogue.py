# mypy: allow-untyped-defs
from __future__ import annotations

import hashlib
from typing import Any

import torch

from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.virtualized import V


class _Expr:
    def __init__(self, source: str):
        self.source = source

    def __str__(self) -> str:
        return self.source


class _EpilogueOps:
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

    def maximum(self, a: Any, b: Any) -> _Expr:
        raise NotImplementedError("FlyDSL GEMM maximum epilogues are not supported yet")

    def minimum(self, a: Any, b: Any) -> _Expr:
        raise NotImplementedError("FlyDSL GEMM minimum epilogues are not supported yet")

    def where(self, condition: Any, a: Any, b: Any) -> _Expr:
        raise NotImplementedError("FlyDSL GEMM where epilogues are not supported yet")

    def to_dtype(
        self, x: Any, dtype: torch.dtype, *, use_compute_types: bool = False
    ) -> _Expr:
        dtype_name = {
            torch.float16: "fx.Float16",
            torch.bfloat16: "fx.BFloat16",
            torch.float32: "fx.Float32",
            torch.int32: "fx.Int32",
            torch.bool: "fx.Boolean",
        }.get(dtype)
        if dtype_name is None:
            raise NotImplementedError(f"unsupported FlyDSL epilogue dtype: {dtype}")
        raise NotImplementedError("FlyDSL GEMM dtype casts are not supported yet")

    convert_element_type = to_dtype

    def __getattr__(self, name: str):
        raise AttributeError(name)


def _arg(value: Any, env: dict[str, Any]) -> Any:
    if hasattr(value, "get_name"):
        name = value.get_name()
        if name in env:
            return env[name]
    if isinstance(value, (int, float, bool, torch.dtype)):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_arg(item, env) for item in value)
    return value


def materialize_flydsl_scheduler_epilogue(
    original_buffer_name: str,
    epilogue_nodes: list[BaseSchedulerNode],
) -> tuple[str, str]:
    if not epilogue_nodes:
        return "", (
            "HAS_EPILOGUE: fx.Constexpr = False\n"
            "EPILOGUE_FN = None\n"
        )

    env: dict[str, Any] = {original_buffer_name: _Expr("acc")}
    handler = _EpilogueOps(original_buffer_name)
    for scheduler_node in epilogue_nodes:
        if scheduler_node.is_reduction():
            raise NotImplementedError("FlyDSL GEMM epilogue reductions unsupported")
        for node in scheduler_node.get_nodes():
            if not isinstance(node.node, ComputedBuffer) or not isinstance(
                node.node.data, Pointwise
            ):
                raise NotImplementedError("FlyDSL GEMM epilogue must be pointwise")
            with V.set_ops_handler(handler), V.set_current_node(node.node):
                result = node.node.data.inner_fn(*node.node.data.inner_fn_args())
            env[node.node.get_name()] = result
            env[scheduler_node.get_name()] = result

    final_name = epilogue_nodes[-1].get_name()
    if final_name not in env:
        raise AssertionError(f"missing final FlyDSL epilogue value {final_name}")

    result = str(env[final_name])
    key = hashlib.sha256(result.encode()).hexdigest()[:16]
    name = f"flydsl_gemm_epilogue_{key}"
    return (
        name,
        f"EPILOGUE_FN = lambda acc: {result}\n"
        "HAS_EPILOGUE: fx.Constexpr = True\n"
    )
