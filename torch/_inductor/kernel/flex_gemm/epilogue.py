# mypy: allow-untyped-defs
import hashlib
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
)
from torch._inductor.virtualized import ops, V
from torch.utils._sympy.value_ranges import ValueRanges


def _cute_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


class FlexGemmCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class FlexGemmCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class FlexGemmCuteDSLKernel:
    def __init__(self) -> None:
        self.body = FlexGemmCuteDSLBody()
        self.cse = FlexGemmCuteDSLCSE()


def output_node(graph_module: torch.fx.GraphModule) -> torch.fx.Node:
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)) and len(output_value) == 1:
        output_value = output_value[0]
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    return output_value


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, (int, float, bool, torch.dtype, torch.device, torch.layout)):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _cute_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _cute_op_name(target)
    if op_name in {"sum", "mean", "amax", "amin", "prod"}:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue reduction: {target}")
    if op_name in ("add", "sub"):
        alpha = kwargs.get("alpha", 1)
        rhs = args[1] if alpha == 1 else ops.mul(args[1], alpha).value
        if op_name == "add":
            return ops.add(args[0], rhs).value
        return ops.sub(args[0], rhs).value
    if kwargs:
        if op_name == "_to_copy":
            return ops.to_dtype(args[0], kwargs["dtype"]).value
        if op_name == "clamp":
            result = args[0]
            min_value = kwargs.get("min", args[1] if len(args) > 1 else None)
            max_value = kwargs.get("max", args[2] if len(args) > 2 else None)
            if min_value is not None:
                result = ops.maximum(result, min_value).value
            if max_value is not None:
                result = ops.minimum(result, max_value).value
            return result
        raise NotImplementedError(
            f"unsupported kwargs for FlexGEMM epilogue op {target}: {kwargs}"
        )
    if op_name == "convert_element_type":
        return ops.to_dtype(args[0], args[1]).value
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    try:
        op = getattr(ops, op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args).value


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> tuple[str, str]:
    gemm = gemm_node(graph_module, gemm_op)
    output = output_node(graph_module)
    kernel = FlexGemmCuteDSLKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }

    with V.set_kernel_handler(kernel), V.set_ops_handler(CuteDSLOpOverrides()):
        for node in graph_module.graph.nodes:
            if node is gemm or node.op in ("placeholder", "output"):
                continue
            with V.set_current_node(node):
                node_args = tuple(_cute_arg(arg, env) for arg in node.args)
                node_kwargs = {
                    key: _cute_arg(value, env) for key, value in node.kwargs.items()
                }
                if node.op in ("call_function", "call_method"):
                    env[node] = _cute_call(node.target, node_args, node_kwargs)
                    continue
                raise NotImplementedError(
                    f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                )

    key = hashlib.sha256(graph_module.code.encode()).hexdigest()[:16]
    name = f"flex_gemm_epilogue_{key}"
    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    return (
        name,
        "import cutlass.cute as cute\n\n"
        f"@cute.jit\ndef {name}(acc):\n"
        f"{body}    return {_cute_arg(output, env)}\n",
    )
