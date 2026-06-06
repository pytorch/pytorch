# mypy: allow-untyped-defs
import hashlib
import operator
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
)
from torch._inductor.virtualized import ops, V
from torch.utils._sympy.value_ranges import ValueRanges


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


def _cute_call_function(
    target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    binary_ops = {
        torch.ops.aten.add.Tensor: ops.add,
        torch.ops.aten.add.Scalar: ops.add,
        operator.add: ops.add,
        torch.ops.aten.sub.Tensor: ops.sub,
        torch.ops.aten.sub.Scalar: ops.sub,
        operator.sub: ops.sub,
        torch.ops.aten.mul.Tensor: ops.mul,
        torch.ops.aten.mul.Scalar: ops.mul,
        operator.mul: ops.mul,
        torch.ops.aten.div.Tensor: ops.truediv,
        torch.ops.aten.div.Scalar: ops.truediv,
        operator.truediv: ops.truediv,
        torch.ops.aten.gt.Scalar: ops.gt,
        torch.ops.aten.gt.Tensor: ops.gt,
        operator.gt: ops.gt,
        torch.ops.aten.ge.Scalar: ops.ge,
        torch.ops.aten.ge.Tensor: ops.ge,
        operator.ge: ops.ge,
        torch.ops.aten.lt.Scalar: ops.lt,
        torch.ops.aten.lt.Tensor: ops.lt,
        operator.lt: ops.lt,
        torch.ops.aten.le.Scalar: ops.le,
        torch.ops.aten.le.Tensor: ops.le,
        operator.le: ops.le,
    }
    if target in binary_ops:
        if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
            alpha = kwargs.get("alpha", 1)
            rhs = args[1] if alpha == 1 else ops.mul(args[1], alpha).value
            return ops.add(args[0], rhs).value
        if target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
            alpha = kwargs.get("alpha", 1)
            rhs = args[1] if alpha == 1 else ops.mul(args[1], alpha).value
            return ops.sub(args[0], rhs).value
        if kwargs:
            raise NotImplementedError(
                f"unsupported kwargs for FlexGEMM epilogue op {target}: {kwargs}"
            )
        return binary_ops[target](*args[:2]).value

    unary_ops = {
        torch.ops.aten.neg.default: ops.neg,
        operator.neg: ops.neg,
        torch.ops.aten.abs.default: ops.abs,
        torch.abs: ops.abs,
        torch.ops.aten.exp.default: ops.exp,
        torch.exp: ops.exp,
        torch.ops.aten.sqrt.default: ops.sqrt,
        torch.sqrt: ops.sqrt,
        torch.ops.aten.sigmoid.default: ops.sigmoid,
        torch.sigmoid: ops.sigmoid,
        torch.ops.aten.tanh.default: ops.tanh,
        torch.tanh: ops.tanh,
        torch.ops.aten.relu.default: ops.relu,
        torch.relu: ops.relu,
    }
    if target in unary_ops:
        return unary_ops[target](args[0]).value

    if target in (torch.where, torch.ops.aten.where.self):
        return ops.where(*args[:3]).value
    if target == torch.ops.aten._to_copy.default:
        return ops.to_dtype(args[0], kwargs["dtype"]).value
    if target == torch.ops.prims.convert_element_type.default:
        return ops.to_dtype(args[0], args[1]).value
    if target in (torch.ops.aten.clamp.default, torch.clamp):
        result = args[0]
        min_value = kwargs.get("min", args[1] if len(args) > 1 else None)
        max_value = kwargs.get("max", args[2] if len(args) > 2 else None)
        if min_value is not None:
            result = ops.maximum(result, min_value).value
        if max_value is not None:
            result = ops.minimum(result, max_value).value
        return result
    raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")


def _cute_call_method(
    target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    method_targets = {
        "relu": torch.ops.aten.relu.default,
        "sigmoid": torch.ops.aten.sigmoid.default,
        "tanh": torch.ops.aten.tanh.default,
        "abs": torch.ops.aten.abs.default,
        "sqrt": torch.ops.aten.sqrt.default,
    }
    if target not in method_targets:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue method: {target}")
    return _cute_call_function(method_targets[target], args, kwargs)


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
                if node.op == "call_function":
                    env[node] = _cute_call_function(
                        node.target,
                        tuple(_cute_arg(arg, env) for arg in node.args),
                        {
                            key: _cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        },
                    )
                    continue
                if node.op == "call_method":
                    env[node] = _cute_call_method(
                        node.target,
                        tuple(_cute_arg(arg, env) for arg in node.args),
                        {
                            key: _cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        },
                    )
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
