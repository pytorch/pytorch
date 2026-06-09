# mypy: allow-untyped-defs
import hashlib
import inspect
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


@dataclass(frozen=True)
class GemmEpilogueOpInfo:
    quack_name: str
    mat1_index: int
    mat2_index: int
    is_batched: bool = False
    supports_quack: bool = True


GEMM_EPILOGUE_OPS = {
    torch.ops.aten.mm.default: GemmEpilogueOpInfo("mm", 0, 1),
    torch.ops.aten.addmm.default: GemmEpilogueOpInfo("addmm", 1, 2),
    torch.ops.aten.bmm.default: GemmEpilogueOpInfo("bmm", 0, 1, is_batched=True),
    torch.ops.aten.baddbmm.default: GemmEpilogueOpInfo(
        "baddbmm", 1, 2, is_batched=True
    ),
    torch.ops.aten._scaled_mm.default: GemmEpilogueOpInfo("scaled_mm", 0, 1),
    torch.ops.aten._grouped_mm.default: GemmEpilogueOpInfo("grouped_mm", 0, 1),
}
_GEMM_EPILOGUE_OP_ALIASES = {
    torch.mm: torch.ops.aten.mm.default,
    torch.addmm: torch.ops.aten.addmm.default,
    torch.bmm: torch.ops.aten.bmm.default,
    torch.baddbmm: torch.ops.aten.baddbmm.default,
    torch._grouped_mm: torch.ops.aten._grouped_mm.default,
}
_SUPPORTED_BACKENDS = {"TRITON", "CUTLASS", "CUTEDSL", "QUACK"}
_SUPPORTED_GEMM_OP_NAMES = "mm/addmm/bmm/baddbmm/_scaled_mm/_grouped_mm"


def _normalize_gemm_epilogue_op(gemm_op: Callable[..., Any]) -> Callable[..., Any]:
    import torch.nn.functional as F

    return {
        **_GEMM_EPILOGUE_OP_ALIASES,
        F.grouped_mm: torch.ops.aten._grouped_mm.default,
    }.get(gemm_op, gemm_op)


def _find_single_quack_gemm_node(graph_module: torch.fx.GraphModule) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if (
            node.op == "call_function"
            and node.target in GEMM_EPILOGUE_OPS
            and GEMM_EPILOGUE_OPS[node.target].supports_quack
        )
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError(
            "QUACK GEMM epilogue backend currently supports one "
            f"{_SUPPORTED_GEMM_OP_NAMES} body"
        )
    return gemm_nodes[0]


class _QuackCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class _QuackCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            CuteDSLCSEVariable,
        )
        from torch.utils._sympy.value_ranges import ValueRanges

        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class _QuackCuteDSLKernel:
    def __init__(self) -> None:
        self.body = _QuackCuteDSLBody()
        self.cse = _QuackCuteDSLCSE()


def _quack_cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, (int, float, bool, torch.dtype, torch.device, torch.layout)):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_quack_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported epilogue constant: {value!r}")


def _quack_cute_call_function(
    target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    from torch._inductor.virtualized import ops

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
        torch.ops.aten.pow.Tensor_Scalar: ops.pow,
        torch.ops.aten.pow.Tensor_Tensor: ops.pow,
        torch.pow: ops.pow,
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
        torch.ops.aten.eq.Scalar: ops.eq,
        torch.ops.aten.eq.Tensor: ops.eq,
        operator.eq: ops.eq,
        torch.ops.aten.ne.Scalar: ops.ne,
        torch.ops.aten.ne.Tensor: ops.ne,
        operator.ne: ops.ne,
        torch.ops.aten.maximum.default: ops.maximum,
        torch.maximum: ops.maximum,
        torch.ops.aten.minimum.default: ops.minimum,
        torch.minimum: ops.minimum,
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
                f"unsupported kwargs for QUACK epilogue op {target}: {kwargs}"
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
        torch.ops.aten.sin.default: ops.sin,
        torch.sin: ops.sin,
        torch.ops.aten.cos.default: ops.cos,
        torch.cos: ops.cos,
        torch.ops.aten.erf.default: ops.erf,
        torch.erf: ops.erf,
        torch.ops.aten.sigmoid.default: ops.sigmoid,
        torch.sigmoid: ops.sigmoid,
        torch.ops.aten.tanh.default: ops.tanh,
        torch.tanh: ops.tanh,
        torch.ops.aten.relu.default: ops.relu,
        torch.relu: ops.relu,
    }
    if target in unary_ops:
        return unary_ops[target](args[0]).value

    if target in (torch.ops.aten.reciprocal.default, torch.reciprocal):
        return ops.truediv(1.0, args[0]).value
    if target in (torch.ops.aten.rsqrt.default, torch.rsqrt):
        return ops.truediv(1.0, ops.sqrt(args[0])).value
    if target in (torch.where, torch.ops.aten.where.self):
        return ops.where(*args[:3]).value
    if target in (torch.tensor, torch.ops.aten.scalar_tensor.default):
        return args[0]
    if target == torch.ops.aten.full.default and args[0] == []:
        return args[1]
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
    if target == torch.ops.aten.clamp_min.default:
        return ops.maximum(args[0], args[1]).value
    if target == torch.ops.aten.clamp_max.default:
        return ops.minimum(args[0], args[1]).value
    if target == torch.nn.functional.leaky_relu:
        negative_slope = kwargs.get(
            "negative_slope", args[1] if len(args) > 1 else 0.01
        )
        return ops.where(
            ops.gt(args[0], 0.0), args[0], ops.mul(args[0], negative_slope)
        ).value
    if target == torch.nn.functional.silu:
        if kwargs:
            raise NotImplementedError(
                f"unsupported kwargs for QUACK epilogue op {target}: {kwargs}"
            )
        return ops.mul(args[0], ops.sigmoid(args[0])).value
    if target == torch._C._nn.gelu:
        approximate = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        if approximate != "none":
            raise NotImplementedError(
                f"unsupported gelu approximate mode for QUACK epilogue: {approximate}"
            )
        return ops.mul(
            ops.mul(args[0], 0.5),
            ops.add(
                1.0,
                ops.erf(ops.truediv(args[0], math.sqrt(2.0))),
            ),
        ).value
    raise NotImplementedError(f"unsupported epilogue op: {target}")


def _quack_cute_epilogue_code(
    graph_module: torch.fx.GraphModule,
) -> tuple[list[str], str]:
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
        ModificationWrapperCuteDSL,
    )
    from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
    from torch._inductor.virtualized import ops, V
    from torch.utils._sympy.value_ranges import ValueRanges

    mm_node = _find_single_quack_gemm_node(graph_module)
    kernel = _QuackCuteDSLKernel()
    handler = ModificationWrapperCuteDSL(kernel, 0, {}, None)
    env: dict[torch.fx.Node, Any] = {
        mm_node: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }
    with V.set_kernel_handler(kernel), V.set_ops_handler(handler):
        for node in graph_module.graph.nodes:
            if node.op in ("placeholder", "output") or node is mm_node:
                continue
            with V.set_current_node(node):
                if node.op == "call_method":
                    arg = _quack_cute_arg(node.args[0], env)
                    if node.target == "relu":
                        env[node] = ops.relu(arg).value
                        continue
                    if node.target == "clamp":
                        result = arg
                        if node.kwargs.get("min") is not None:
                            result = ops.maximum(result, node.kwargs["min"]).value
                        if node.kwargs.get("max") is not None:
                            result = ops.minimum(result, node.kwargs["max"]).value
                        env[node] = result
                        continue
                if node.op == "call_function":
                    env[node] = _quack_cute_call_function(
                        node.target,
                        tuple(_quack_cute_arg(arg, env) for arg in node.args),
                        {
                            key: _quack_cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        },
                    )
                    continue
            raise NotImplementedError(
                f"unsupported epilogue node: {node.format_node()}"
            )

    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("QUACK GEMM epilogue backend expects one output")
    return kernel.body.lines, str(_quack_cute_arg(_unwrap_output(output_nodes[0]), env))


def _render_quack_gemm_epilogue(
    epilogue_name: str, body_lines: list[str], result: str
) -> str:
    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.kernel.mm_common import load_kernel_template

    template = KernelTemplate._template_from_string(
        load_kernel_template("quack_gemm_epilogue")
    )
    if template is None:
        raise ImportError("jinja2 is required to render QuACK GEMM epilogue templates")
    return template.render(
        epilogue_name=epilogue_name, body_lines=body_lines, result=result
    )


def materialize_quack_epilogue(graph_module: torch.fx.GraphModule) -> tuple[str, str]:
    lines, result = _quack_cute_epilogue_code(graph_module)
    key = (
        "flex_gemm_quack_epilogue_"
        + hashlib.sha256(graph_module.code.encode()).hexdigest()[:16]
    )
    return key, _render_quack_gemm_epilogue(key, lines, result)


def _unwrap_output(node: torch.fx.Node) -> Any:
    value = node.args[0]
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value


def is_mm_relu_body(graph_module: torch.fx.GraphModule) -> bool:
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    mm_nodes = [
        node
        for node in nodes
        if node.op == "call_function" and node.target == torch.ops.aten.mm.default
    ]
    if len(placeholders) != 2 or len(outputs) != 1 or len(mm_nodes) != 1:
        return False

    mm_node = mm_nodes[0]
    relu_nodes = [
        user
        for user in mm_node.users
        if (
            (user.op == "call_method" and user.target == "relu")
            or (
                user.op == "call_function"
                and user.target in (torch.ops.aten.relu.default, torch.relu)
            )
        )
        and user.args[0] is mm_node
    ]
    return len(relu_nodes) == 1 and _unwrap_output(outputs[0]) is relu_nodes[0]


class GemmEpilogueFusion(HigherOrderOperator):
    def __init__(self):
        super().__init__("gemm_epilogue_fusion")

    def __call__(
        self,
        gemm_op: torch._ops.OpOverload,
        body_fn: Callable[..., Any],
        gemm_args: tuple[Any, ...],
        gemm_kwargs: dict[str, Any],
        kernel_options: dict[str, Any],
    ):
        if gemm_op not in GEMM_EPILOGUE_OPS:
            raise RuntimeError(f"unsupported GEMM op for epilogue fusion: {gemm_op}")

        if not isinstance(gemm_args, tuple):
            gemm_args = tuple(gemm_args)

        if not all(
            isinstance(
                t,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    torch.SymBool,
                    int,
                    float,
                    bool,
                ),
            )
            for t in gemm_args
        ):
            raise RuntimeError(
                "gemm_args must be a tuple of tensors, SymInts, SymFloats, "
                f"SymBools, ints, floats, or bools, got {gemm_args}"
            )

        if not isinstance(gemm_kwargs, dict):
            raise RuntimeError(f"gemm_kwargs must be a dict, got {type(gemm_kwargs)}")

        if not isinstance(kernel_options, dict):
            raise RuntimeError(
                f"kernel_options must be a dict, got {type(kernel_options)}"
            )

        kernel_options = {"backend": "TRITON", "SPLIT_K": False, **kernel_options}
        backend = kernel_options.get("backend", "TRITON")
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported GEMM epilogue backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )

        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


_gemm_epilogue_fusion = GemmEpilogueFusion()


def gemm_epilogue_fusion(
    gemm_op: Callable[..., Any],
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
    kernel_options: dict[str, Any] | None = None,
):
    if gemm_kwargs is None:
        gemm_kwargs = {}
    if kernel_options is None:
        kernel_options = {"backend": "TRITON", "SPLIT_K": False}
    gemm_op = _normalize_gemm_epilogue_op(gemm_op)

    def body_fn(*args, **body_kwargs):
        return epilogue_fn(gemm_op(*args, **body_kwargs))

    body_fn._gemm_epilogue_accepts_kwargs = True
    return _gemm_epilogue_fusion(
        gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
    )


def mm_epilogue(
    input: Any,
    mat2: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    kernel_options: dict[str, Any] | None = None,
):
    return gemm_epilogue_fusion(
        torch.ops.aten.mm.default,
        (input, mat2),
        epilogue_fn,
        kernel_options=kernel_options,
    )


def addmm_epilogue(
    input: Any,
    mat1: Any,
    mat2: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    kernel_options: dict[str, Any] | None = None,
):
    return gemm_epilogue_fusion(
        torch.ops.aten.addmm.default,
        (input, mat1, mat2),
        epilogue_fn,
        gemm_kwargs={"beta": beta, "alpha": alpha},
        kernel_options=kernel_options,
    )


def bmm_epilogue(
    input: Any,
    mat2: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    kernel_options: dict[str, Any] | None = None,
):
    return gemm_epilogue_fusion(
        torch.ops.aten.bmm.default,
        (input, mat2),
        epilogue_fn,
        kernel_options=kernel_options,
    )


def baddbmm_epilogue(
    input: Any,
    batch1: Any,
    batch2: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
    kernel_options: dict[str, Any] | None = None,
):
    return gemm_epilogue_fusion(
        torch.ops.aten.baddbmm.default,
        (input, batch1, batch2),
        epilogue_fn,
        gemm_kwargs={"beta": beta, "alpha": alpha},
        kernel_options=kernel_options,
    )


def grouped_mm_epilogue(
    mat_a: Any,
    mat_b: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    offs: Any | None = None,
    bias: Any | None = None,
    out_dtype: torch.dtype | None = None,
    kernel_options: dict[str, Any] | None = None,
):
    gemm_args = [mat_a, mat_b]
    offs_index = None
    bias_index = None
    if offs is not None:
        offs_index = len(gemm_args)
        gemm_args.append(offs)
    if bias is not None:
        bias_index = len(gemm_args)
        gemm_args.append(bias)

    def body_fn(*body_args):
        body_offs = None if offs_index is None else body_args[offs_index]
        body_bias = None if bias_index is None else body_args[bias_index]
        return epilogue_fn(
            torch.ops.aten._grouped_mm.default(
                body_args[0],
                body_args[1],
                offs=body_offs,
                bias=body_bias,
                out_dtype=out_dtype,
            )
        )

    return _gemm_epilogue_fusion(
        torch.ops.aten._grouped_mm.default,
        body_fn,
        tuple(gemm_args),
        {},
        {"backend": "TRITON"} if kernel_options is None else kernel_options,
    )


def matmul_epilogue(
    input: Any,
    other: Any,
    epilogue_fn: Callable[[Any], Any],
    *,
    kernel_options: dict[str, Any] | None = None,
):
    """Apply an epilogue to GEMM-shaped matmul cases only.

    This helper intentionally does not implement vector inputs or broadcasted
    matmul. Use the lower-level helpers for explicit control over supported
    `mm` and `bmm` cases.
    """
    if input.dim() == 2 and other.dim() == 2:
        return mm_epilogue(input, other, epilogue_fn, kernel_options=kernel_options)
    if input.dim() == 3 and other.dim() == 3:
        return bmm_epilogue(input, other, epilogue_fn, kernel_options=kernel_options)
    raise NotImplementedError(
        "matmul_epilogue currently supports only 2D mm and 3D bmm inputs"
    )


def _body_accepts_kwargs(body_fn, kwargs) -> bool:
    if not kwargs or isinstance(body_fn, torch.fx.GraphModule):
        return False
    if getattr(body_fn, "_gemm_epilogue_accepts_kwargs", False):
        return True
    signature = inspect.signature(body_fn)
    return all(key in signature.parameters for key in kwargs)


def _call_gemm_epilogue_body(body_fn, args, kwargs):
    if _body_accepts_kwargs(body_fn, kwargs):
        return body_fn(*args, **kwargs)
    return body_fn(*args)


@_gemm_epilogue_fusion.py_impl(DispatchKey.CompositeExplicitAutograd)
def gemm_epilogue_fusion_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    return _call_gemm_epilogue_body(body_fn, args, kwargs)


_gemm_epilogue_fusion.py_autograd_impl(
    autograd_not_implemented(_gemm_epilogue_fusion, deferred_error=True)
)


@_gemm_epilogue_fusion.py_impl(FakeTensorMode)
def gemm_epilogue_fusion_fake_tensor_mode(
    mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    flat_args = pytree.tree_leaves(args)
    with mode:
        return _call_gemm_epilogue_body(body_fn, flat_args, kwargs)


@_gemm_epilogue_fusion.py_functionalize_impl
def gemm_epilogue_fusion_functionalize(
    ctx, gemm_op, body_fn, args, kwargs, kernel_options
):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_kernel_options = ctx.unwrap_tensors(kernel_options)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            body_fn, unwrapped_args, "gemm_epilogue_fusion", pre_dispatch
        )

        outputs = _gemm_epilogue_fusion(
            gemm_op,
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_kernel_options,
        )
        return ctx.wrap_tensors(outputs)


@_gemm_epilogue_fusion.py_impl(ProxyTorchDispatchMode)
def gemm_epilogue_fusion_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    if proxy_mode.enable_tracing:
        flat_args = tuple(pytree.tree_leaves(args))

        def tracing_body_fn(*flat_body_args):
            return _call_gemm_epilogue_body(body_fn, flat_body_args, kwargs)

        body_graph = reenter_make_fx(tracing_body_fn)(*flat_args)
        _, body_graph_name = unique_graph_id(
            proxy_mode, prefix="gemm_epilogue_fusion_body_graph"
        )
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            _gemm_epilogue_fusion,
            proxy_args,
            {},
            name="gemm_epilogue_fusion",
        )
        out = _call_gemm_epilogue_body(body_fn, flat_args, kwargs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )
    return _gemm_epilogue_fusion(gemm_op, body_fn, args, kwargs, kernel_options)
