# mypy: allow-untyped-defs
import dataclasses
import hashlib
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    upcast_compute_type,
)
from torch._inductor.virtualized import V
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


class FlexGemmCuteDSLOpOverrides(CuteDSLOpOverrides):
    # Aten add/sub carry alpha as schema sugar; CuTeDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                "unsupported kwargs for FlexGEMM epilogue op _to_copy: "
                f"{unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan:
    """Classify the FlexGEMM body output into a main result and aux returns."""

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()


def output_plan(
    graph_module: torch.fx.GraphModule,
) -> FlexGemmOutputPlan:
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            return FlexGemmOutputPlan(output, tuple(aux_outputs))
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    return FlexGemmOutputPlan(output_value)


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
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _cute_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _cute_op_name(target)
    if op_name in {"sum", "mean", "amax", "amin", "prod"}:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue reduction: {target}")
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Build the generated CuTeDSL epilogue callable from the traced FX body."""
    gemm = gemm_node(graph_module, gemm_op)
    outputs = output_plan(graph_module)
    kernel = FlexGemmCuteDSLKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }
    with V.set_kernel_handler(kernel), V.set_ops_handler(FlexGemmCuteDSLOpOverrides()):
        for index, node in enumerate(epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    env[node], logical_dtype, use_compute_types=False
                )

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

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    aux_args = [f"aux{index}" for index in range(len(epilogue_arg_placeholders))]
    epilogue_params = ", ".join(["acc", *aux_args])
    result = _cute_arg(outputs.output, env)
    if outputs.aux_outputs:
        aux_results = [_cute_arg(aux_output, env) for aux_output in outputs.aux_outputs]
        result = f"({', '.join(str(item) for item in (result, *aux_results))})"
    key_payload = f"{graph_module.code}\n{body}\nreturn {result}"
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flex_gemm_epilogue_{key}"
    return (
        name,
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "import operator\n"
        "from cutlass._mlir.dialects import math as mlir_math\n\n"
        f"@cute.jit\ndef {name}({epilogue_params}):\n"
        f"{body}    return {result}\n",
    )
