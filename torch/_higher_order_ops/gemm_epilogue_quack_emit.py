# mypy: allow-untyped-defs
import hashlib
import math
import operator
from typing import Any

import torch
import torch.utils._pytree as pytree

from torch._higher_order_ops.gemm_epilogue_quack import (
    QuackAuxOutputInfo,
    QuackGroupedTensorSSAInfo,
    QuackLocalReduceInfo,
    QuackMainOutputTransformInfo,
    QuackTensorSSAReduceMatch,
    QuackViewMatch,
    analyze_output,
    find_single_gemm_node,
    is_concat_half_n_shape,
    is_n_group_shape,
    is_nonnegative_expr,
    is_same_fragment_n_group_shape,
    is_scalar_one,
    is_scalar_value,
    match_grouped_n_split,
    match_grouped_n_select,
    match_mul_scalar,
    match_negated_node,
    match_tensorssa_reduce,
    method_clamp_bounds,
    match_view_or_reshape,
    normalize_shape,
    fx_equivalent,
    grouped_n_fragment_shape,
    tensorssa_grouped_n_shape,
    QUACK_TENSORSSA_FRAGMENT_N,
    QUACK_TENSORSSA_REDUCTIONS,
    unwrap_output,
)


class QuackCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class QuackCuteDSLCSE:
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


class QuackCuteDSLKernel:
    def __init__(self) -> None:
        self.body = QuackCuteDSLBody()
        self.cse = QuackCuteDSLCSE()


def cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, (int, float, bool, torch.dtype, torch.device, torch.layout)):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported epilogue constant: {value!r}")


def cute_call_function(
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
        if kwargs["dtype"] == torch.float32 and not hasattr(args[0], "to"):
            return args[0]
        return ops.to_dtype(args[0], kwargs["dtype"]).value
    if target == torch.ops.prims.convert_element_type.default:
        if args[1] == torch.float32 and not hasattr(args[0], "to"):
            return args[0]
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
    if target in (torch.nn.functional.silu, torch.ops.aten.silu.default):
        if kwargs:
            raise NotImplementedError(
                f"unsupported kwargs for QUACK epilogue op {target}: {kwargs}"
            )
        # Prefer the tanh identity over sigmoid/exp/div for QUACK TensorSSA:
        #   silu(x) = h * tanh(h) + h, h = 0.5 * x
        # The explicit flex_gemm::silu_tanh op lowers this with fastmath=True;
        # plain F.silu still uses the generic ops handler here, so use tanh form
        # to avoid the much slower exp2+divide path.
        half = ops.mul(args[0], 0.5).value
        return ops.add(ops.mul(half, ops.tanh(half)).value, half).value
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

def emit_tensorssa_expr(
    kernel: QuackCuteDSLKernel,
    expr: str,
    *,
    like: Any | None = None,
    dtype: torch.dtype | None = None,
    shape: Any | None = None,
) -> Any:
    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=dtype if dtype is not None else getattr(like, "dtype", None),
        shape=shape if shape is not None else getattr(like, "shape", None),
    )


def emit_tensorssa_reshape(
    kernel: QuackCuteDSLKernel, value: Any, shape: str
) -> Any:
    return emit_tensorssa_expr(kernel, f"{value}.reshape({shape})", like=value)


def emit_tensorssa_reduce(
    kernel: QuackCuteDSLKernel,
    value: Any,
    *,
    op: str,
    init_val: str,
    reduction_profile: str,
) -> Any:
    return emit_tensorssa_expr(
        kernel,
        f"{value}.reduce({op}, init_val={init_val}, reduction_profile={reduction_profile})",
        like=value,
    )


def emit_tensorssa_broadcast_to(
    kernel: QuackCuteDSLKernel, value: Any, shape: str
) -> Any:
    return emit_tensorssa_expr(kernel, f"{value}.broadcast_to({shape})", like=value)

def lower_view_or_reshape_node(
    node: torch.fx.Node,
    view_match: QuackViewMatch,
    env: dict[torch.fx.Node, Any],
    kernel: QuackCuteDSLKernel,
    mm_node: torch.fx.Node,
) -> Any:
    source = cute_arg(view_match.base, env)
    shape = normalize_shape(view_match.shape)
    group_shape = grouped_n_fragment_shape(shape)
    if is_n_group_shape(group_shape) or is_concat_half_n_shape(shape):
        if is_concat_half_n_shape(shape):
            group_size = 2
        else:
            if not is_same_fragment_n_group_shape(group_shape):
                raise NotImplementedError(
                    "QUACK reductions feeding the main output currently require a "
                    "power-of-two group size that divides the same-fragment N width 32; "
                    "aux reductions can use other static groups"
                )
            group_size = group_shape[-1]
        return emit_tensorssa_reshape(kernel, source, tensorssa_grouped_n_shape(group_size))
    mm_meta = mm_node.meta.get("val")
    if (
        mm_meta is not None
        and isinstance(shape, (list, tuple))
        and tuple(shape) == tuple(mm_meta.shape)
    ):
        return emit_tensorssa_reshape(kernel, source, f"{env[mm_node]}.shape")
    raise NotImplementedError(f"unsupported QUACK epilogue view/reshape: {node.format_node()}")


def lower_grouped_n_select_node(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: QuackCuteDSLKernel,
    grouped_tensors: dict[torch.fx.Node, QuackGroupedTensorSSAInfo],
) -> Any | None:
    index = None
    if (
        node.op == "call_function"
        and node.target == torch.ops.aten.select.int
        and len(node.args) >= 3
        and isinstance(node.args[2], int)
    ):
        view_node = node.args[0]
        view = match_view_or_reshape(view_node)
        view_shape = normalize_shape(view.shape) if view is not None else None
        dim = node.args[1]
        is_group_dim = dim == -1 or (
            isinstance(view_shape, (list, tuple)) and dim == len(view_shape) - 1
        )
        if is_concat_half_n_shape(view_shape) and dim == 1:
            is_group_dim = True
        if not is_group_dim:
            return None
        index = node.args[2]
    elif (
        node.op == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and isinstance(node.args[1], int)
    ):
        view_node = node.args[0]
        index = node.args[1]
    else:
        return None
    info = grouped_tensors.get(view_node)
    if info is None or not (0 <= index < info.group_size):
        return None
    source = cute_arg(view_node, env)
    return emit_tensorssa_expr(
        kernel,
        f"{source}[((0, {index}, None), None, None)]",
        like=source,
    )


def lower_tensorssa_reduce_node(
    reduce_match: QuackTensorSSAReduceMatch,
    env: dict[torch.fx.Node, Any],
    kernel: QuackCuteDSLKernel,
    grouped_tensors: dict[torch.fx.Node, QuackGroupedTensorSSAInfo],
) -> Any:
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        raise NotImplementedError(
            f"unsupported QUACK epilogue reduction: {reduce_match.node.format_node()}"
        )
    info = (
        grouped_tensors.get(reduce_match.input_node)
        if isinstance(reduce_match.input_node, torch.fx.Node)
        else None
    )
    if info is None:
        raise NotImplementedError(
            f"unsupported QUACK epilogue reduction input: {reduce_match.node.format_node()}"
        )
    desc = QUACK_TENSORSSA_REDUCTIONS[reduce_match.kind]
    if desc.requires_nonnegative and not info.nonnegative:
        raise NotImplementedError(
            "QUACK amax TensorSSA reduction currently requires an abs/nonnegative input"
        )
    source = cute_arg(reduce_match.input_node, env)
    reduced = emit_tensorssa_reduce(
        kernel,
        source,
        op=desc.cute_op,
        init_val=desc.init_val,
        reduction_profile="((None, 1, None), 1, 1)",
    )
    if bool(reduce_match.keepdim):
        return emit_tensorssa_broadcast_to(
            kernel,
            emit_tensorssa_reshape(kernel, reduced, info.keepdim_reshape),
            f"{source}.shape",
        )
    return reduced

def propagate_grouped_tensorssa_info(
    node: torch.fx.Node,
    grouped_tensors: dict[torch.fx.Node, QuackGroupedTensorSSAInfo],
) -> None:
    input_infos = [
        grouped_tensors[arg]
        for arg in pytree.tree_leaves((node.args, node.kwargs))
        if isinstance(arg, torch.fx.Node) and arg in grouped_tensors
    ]
    if not input_infos:
        return
    first = input_infos[0]
    if any(
        info.group_size != first.group_size
        or info.groups_per_fragment != first.groups_per_fragment
        for info in input_infos
    ):
        return
    is_abs = (
        (node.op == "call_function" and node.target in (torch.ops.aten.abs.default, torch.abs))
        or (node.op == "call_method" and node.target == "abs")
    )
    grouped_tensors[node] = QuackGroupedTensorSSAInfo(
        group_size=first.group_size,
        groups_per_fragment=first.groups_per_fragment,
        nonnegative=is_abs or all(info.nonnegative for info in input_infos),
    )

def match_scaled_sigmoid_source(node: Any) -> tuple[torch.fx.Node, float] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.sigmoid.default,
        torch.sigmoid,
    ):
        return None
    scale_node = node.args[0]
    if not isinstance(scale_node, torch.fx.Node) or scale_node.op != "call_function":
        return None
    if scale_node.target not in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        return None
    lhs, rhs = scale_node.args[:2]
    if isinstance(lhs, torch.fx.Node) and isinstance(rhs, (int, float)):
        return lhs, float(rhs)
    if isinstance(rhs, torch.fx.Node) and isinstance(lhs, (int, float)):
        return rhs, float(lhs)
    return None


def match_fast_quick_gelu(node: torch.fx.Node) -> tuple[torch.fx.Node, float] | None:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        return None
    lhs, rhs = node.args[:2]
    for source, sigmoid_node in ((lhs, rhs), (rhs, lhs)):
        if not isinstance(source, torch.fx.Node):
            continue
        sigmoid_match = match_scaled_sigmoid_source(sigmoid_node)
        if sigmoid_match is None:
            continue
        sigmoid_source, alpha = sigmoid_match
        if fx_equivalent(source, sigmoid_source):
            return source, alpha
    return None


def is_fast_quick_gelu_intermediate(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in (torch.ops.aten.sigmoid.default, torch.sigmoid):
        return any(match_fast_quick_gelu(user) is not None for user in node.users)
    if node.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if any(
            sigmoid_user.op == "call_function"
            and sigmoid_user.target in (torch.ops.aten.sigmoid.default, torch.sigmoid)
            and is_fast_quick_gelu_intermediate(sigmoid_user)
            for sigmoid_user in node.users
        ):
            return True
        return False
    if len(node.users) == 1:
        user = next(iter(node.users))
        if is_fast_quick_gelu_intermediate(user):
            return True
    return False


def match_fast_silu_source(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        return None
    source, denominator = node.args[:2]
    if not isinstance(source, torch.fx.Node) or not isinstance(denominator, torch.fx.Node):
        return None
    if denominator.op != "call_function" or denominator.target not in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        operator.add,
    ):
        return None
    lhs, rhs = denominator.args[:2]
    exp_node = rhs if is_scalar_one(lhs) else lhs if is_scalar_one(rhs) else None
    if not isinstance(exp_node, torch.fx.Node):
        return None
    if exp_node.op != "call_function" or exp_node.target not in (
        torch.ops.aten.exp.default,
        torch.exp,
    ):
        return None
    if match_negated_node(exp_node.args[0], source):
        return source
    return None


def match_fast_gelu_source(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or node.target not in (
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        operator.mul,
    ):
        return None
    lhs, rhs = node.args[:2]
    for half_mul, erf_add in ((lhs, rhs), (rhs, lhs)):
        source = match_mul_scalar(half_mul, 0.5)
        if source is None or not isinstance(erf_add, torch.fx.Node):
            continue
        if erf_add.op != "call_function" or erf_add.target not in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
            operator.add,
        ):
            continue
        add_lhs, add_rhs = erf_add.args[:2]
        erf_node = add_rhs if is_scalar_one(add_lhs) else add_lhs if is_scalar_one(add_rhs) else None
        if not isinstance(erf_node, torch.fx.Node):
            continue
        if erf_node.op != "call_function" or erf_node.target not in (
            torch.ops.aten.erf.default,
            torch.erf,
        ):
            continue
        erf_arg_source = match_mul_scalar(erf_node.args[0], 1 / math.sqrt(2.0))
        if erf_arg_source is source:
            return source
    return None


def is_fast_gelu_intermediate(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if any(
            erf_user.op == "call_function"
            and erf_user.target in (torch.ops.aten.erf.default, torch.erf)
            and is_fast_gelu_intermediate(erf_user)
            for erf_user in node.users
        ):
            return True
        return any(match_fast_gelu_source(user) is not None for user in node.users)
    if node.target in (torch.ops.aten.erf.default, torch.erf):
        return any(
            add_user.op == "call_function"
            and add_user.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, operator.add)
            and is_fast_gelu_intermediate(add_user)
            for add_user in node.users
        )
    if node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, operator.add):
        return any(match_fast_gelu_source(user) is not None for user in node.users)
    return False


def is_fast_silu_intermediate(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in (torch.ops.aten.neg.default, operator.neg):
        source = node.args[0]
        return any(
            exp_user.op == "call_function"
            and exp_user.target in (torch.ops.aten.exp.default, torch.exp)
            and is_fast_silu_intermediate(exp_user)
            for exp_user in node.users
        )
    if node.target in (torch.ops.aten.exp.default, torch.exp):
        return any(
            add_user.op == "call_function"
            and add_user.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, operator.add)
            and is_fast_silu_intermediate(add_user)
            for add_user in node.users
        )
    if node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, operator.add):
        return any(
            div_user.op == "call_function"
            and div_user.target
            in (torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar, operator.truediv)
            and len(div_user.args) > 1
            and div_user.args[1] is node
            and match_fast_silu_source(div_user) is not None
            for div_user in node.users
        )
    return False


def emit_fast_unary(
    kernel: QuackCuteDSLKernel,
    source: Any,
    expr: str,
) -> Any:
    return emit_tensorssa_expr(kernel, expr.format(x=source), like=source)


def emit_fast_sigmoid(kernel: QuackCuteDSLKernel, source: Any) -> Any:
    return emit_fast_unary(
        kernel,
        source,
        "(cute.full_like({x}, 1.0) / (cute.full_like({x}, 1.0) + cute.math.exp(-{x}, fastmath=True)))",
    )


def emit_fast_softplus(kernel: QuackCuteDSLKernel, source: Any) -> Any:
    return emit_tensorssa_expr(
        kernel,
        "cute.where({x} > cute.full_like({x}, 20.0), {x}, "
        "cute.math.log(cute.math.exp({x}, fastmath=True) + cute.full_like({x}, 1.0), fastmath=True))".format(
            x=source
        ),
        like=source,
    )


def emit_fast_silu(kernel: QuackCuteDSLKernel, source: Any) -> Any:
    half = emit_tensorssa_expr(
        kernel,
        f"({source} * cute.full_like({source}, 0.5))",
        like=source,
    )
    return emit_tensorssa_expr(
        kernel,
        f"({half} * cute.math.tanh({half}, fastmath=True) + {half})",
        like=source,
    )


def emit_fast_quick_gelu(
    kernel: QuackCuteDSLKernel, source: Any, alpha: float
) -> Any:
    half_alpha = 0.5 * alpha
    half_source = emit_tensorssa_expr(
        kernel,
        f"({source} * cute.full_like({source}, 0.5))",
        like=source,
    )
    tanh_arg = emit_tensorssa_expr(
        kernel,
        f"({source} * cute.full_like({source}, {half_alpha!r}))",
        like=source,
    )
    return emit_tensorssa_expr(
        kernel,
        f"({half_source} * cute.math.tanh({tanh_arg}, fastmath=True) + {half_source})",
        like=source,
    )


def emit_fast_gelu(kernel: QuackCuteDSLKernel, source: Any) -> Any:
    return emit_tensorssa_expr(
        kernel,
        "(cute.full_like({x}, 0.5) * {x} * "
        "(cute.full_like({x}, 1.0) + cute.math.tanh("
        "{x} * (cute.full_like({x}, 0.7978845608028654) + "
        "cute.full_like({x}, 0.035677408136300125) * {x} * {x}), "
        "fastmath=True)))".format(x=source),
        like=source,
    )


def compile_pointwise_nodes(
    graph_module: torch.fx.GraphModule,
    mm_node: torch.fx.Node,
    skip_nodes: frozenset[torch.fx.Node],
    env: dict[torch.fx.Node, Any],
    kernel: QuackCuteDSLKernel,
    handler: Any,
    *,
    fast_math: bool = False,
) -> None:
    from torch._inductor.virtualized import ops, V

    grouped_tensors: dict[torch.fx.Node, QuackGroupedTensorSSAInfo] = {}
    with V.set_kernel_handler(kernel), V.set_ops_handler(handler):
        for node in graph_module.graph.nodes:
            if (
                node.op in ("placeholder", "output")
                or node is mm_node
                or node in skip_nodes
            ):
                continue
            with V.set_current_node(node):
                view_match = match_view_or_reshape(node)
                if view_match is not None:
                    shape = normalize_shape(view_match.shape)
                    env[node] = lower_view_or_reshape_node(
                        node, view_match, env, kernel, mm_node
                    )
                    group_shape = grouped_n_fragment_shape(shape)
                    if is_same_fragment_n_group_shape(
                        group_shape
                    ) or is_concat_half_n_shape(shape):
                        group_size = (
                            2 if is_concat_half_n_shape(shape) else group_shape[-1]
                        )
                        base_info = grouped_tensors.get(view_match.base)
                        grouped_tensors[node] = QuackGroupedTensorSSAInfo(
                            group_size=group_size,
                            groups_per_fragment=(
                                QUACK_TENSORSSA_FRAGMENT_N // group_size
                            ),
                            nonnegative=(
                                base_info.nonnegative
                                if base_info is not None
                                else is_nonnegative_expr(view_match.base)
                            ),
                        )
                    continue
                split = match_grouped_n_split(node, mm_node)
                if split is not None:
                    _split_node, group_size, _concat_layout = split
                    source = cute_arg(mm_node, env)
                    env[node] = emit_tensorssa_reshape(
                        kernel, source, tensorssa_grouped_n_shape(group_size)
                    )
                    grouped_tensors[node] = QuackGroupedTensorSSAInfo(
                        group_size=group_size,
                        groups_per_fragment=(
                            QUACK_TENSORSSA_FRAGMENT_N // group_size
                        ),
                        nonnegative=is_nonnegative_expr(mm_node),
                    )
                    continue
                select_value = lower_grouped_n_select_node(
                    node, env, kernel, grouped_tensors
                )
                if select_value is not None:
                    env[node] = select_value
                    continue
                reduce_match = match_tensorssa_reduce(node)
                if reduce_match is not None:
                    env[node] = lower_tensorssa_reduce_node(
                        reduce_match, env, kernel, grouped_tensors
                    )
                    continue
                if node.op == "call_method":
                    arg = cute_arg(node.args[0], env)
                    if node.target == "relu":
                        env[node] = ops.relu(arg).value
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == "abs":
                        input_info = grouped_tensors.get(node.args[0])
                        if input_info is not None and input_info.nonnegative:
                            env[node] = arg
                            grouped_tensors[node] = input_info
                        else:
                            env[node] = ops.abs(arg).value
                            propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == "clamp":
                        min_value, max_value = method_clamp_bounds(node)
                        result = arg
                        if min_value is not None:
                            result = ops.maximum(result, min_value).value
                        if max_value is not None:
                            result = ops.minimum(result, max_value).value
                        env[node] = result
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                if node.op == "call_function":
                    if fast_math and is_fast_quick_gelu_intermediate(node):
                        continue
                    if fast_math and is_fast_silu_intermediate(node):
                        continue
                    if fast_math and is_fast_gelu_intermediate(node):
                        continue
                    if fast_math:
                        quick_gelu_match = match_fast_quick_gelu(node)
                        if quick_gelu_match is not None:
                            quick_gelu_source, alpha = quick_gelu_match
                            source = cute_arg(quick_gelu_source, env)
                            env[node] = emit_fast_quick_gelu(kernel, source, alpha)
                            propagate_grouped_tensorssa_info(node, grouped_tensors)
                            continue
                        silu_source = match_fast_silu_source(node)
                        if silu_source is not None:
                            source = cute_arg(silu_source, env)
                            env[node] = emit_fast_silu(kernel, source)
                            propagate_grouped_tensorssa_info(node, grouped_tensors)
                            continue
                        gelu_source = match_fast_gelu_source(node)
                        if gelu_source is not None:
                            source = cute_arg(gelu_source, env)
                            env[node] = emit_fast_gelu(kernel, source)
                            propagate_grouped_tensorssa_info(node, grouped_tensors)
                            continue
                    if fast_math and node.target in (
                        torch.nn.functional.silu,
                        torch.ops.aten.silu.default,
                    ):
                        if node.kwargs:
                            raise NotImplementedError(
                                f"unsupported kwargs for QUACK fast_math silu epilogue op {node.target}: {node.kwargs}"
                            )
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_silu(kernel, source)
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch.ops.aten.tanh.default,
                        torch.tanh,
                    ):
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_unary(
                            kernel, source, "cute.math.tanh({x}, fastmath=True)"
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch.ops.aten.exp.default,
                        torch.exp,
                    ):
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_unary(
                            kernel, source, "cute.math.exp({x}, fastmath=True)"
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch.ops.aten.log.default,
                        torch.log,
                    ):
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_unary(
                            kernel, source, "cute.math.log({x}, fastmath=True)"
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch.ops.aten.log1p.default,
                        torch.log1p,
                    ):
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_unary(
                            kernel,
                            source,
                            "cute.math.log(cute.full_like({x}, 1.0) + {x}, fastmath=True)",
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target == torch.ops.aten.softplus.default:
                        beta = node.args[1] if len(node.args) > 1 else node.kwargs.get("beta", 1)
                        threshold = (
                            node.args[2] if len(node.args) > 2 else node.kwargs.get("threshold", 20)
                        )
                        if beta != 1 or threshold != 20:
                            raise NotImplementedError(
                                "QUACK fast_math softplus currently supports only beta=1, threshold=20"
                            )
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_softplus(kernel, source)
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch.ops.aten.sigmoid.default,
                        torch.sigmoid,
                    ):
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_sigmoid(kernel, source)
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if fast_math and node.target in (
                        torch._C._nn.gelu,
                        torch.ops.aten.gelu.default,
                    ):
                        approximate = node.kwargs.get(
                            "approximate", node.args[1] if len(node.args) > 1 else "none"
                        )
                        if approximate not in ("none", "tanh"):
                            raise NotImplementedError(
                                f"unsupported gelu approximate mode for QUACK fast_math epilogue: {approximate}"
                            )
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_fast_gelu(kernel, source)
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target in (torch.ops.aten.abs.default, torch.abs):
                        input_info = grouped_tensors.get(node.args[0])
                        if input_info is not None and input_info.nonnegative:
                            env[node] = cute_arg(node.args[0], env)
                            grouped_tensors[node] = input_info
                            continue
                    if node.target == torch.ops.flex_gemm.silu_tanh.default:
                        source = cute_arg(node.args[0], env)
                        half = emit_tensorssa_expr(
                            kernel,
                            f"({source} * cute.full_like({source}, 0.5))",
                            like=source,
                        )
                        env[node] = emit_tensorssa_expr(
                            kernel,
                            f"({half} * cute.math.tanh({half}, fastmath=True) + {half})",
                            like=source,
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == torch.ops.flex_gemm.tanh_fast.default:
                        source = cute_arg(node.args[0], env)
                        env[node] = emit_tensorssa_expr(
                            kernel,
                            f"cute.math.tanh({source}, fastmath=True)",
                            like=source,
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == torch.ops.flex_gemm.mx_e8m0_scale.default:
                        source = cute_arg(node.args[0], env)
                        max_power = node.args[1] if len(node.args) > 1 else node.kwargs.get("max_power", 8)
                        scale_exp = f"(cute.math.floor(cute.math.log2({source})) - {max_power})"
                        env[node] = emit_tensorssa_expr(
                            kernel,
                            "cute.math.exp2("
                            f"cute.where({scale_exp} < -127.0, -127.0, "
                            f"cute.where({scale_exp} > 128.0, 128.0, {scale_exp}))"
                            ")",
                            like=source,
                        )
                        propagate_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == torch.ops.flex_gemm.nvfp4_e4m3_scale.default:
                        raise NotImplementedError(
                            "QUACK nvfp4_e4m3_scale feeding the main output requires "
                            "E4M3 rounding semantics and is not implemented yet"
                        )
                    env[node] = cute_call_function(
                        node.target,
                        tuple(cute_arg(arg, env) for arg in node.args),
                        {
                            key: cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        },
                    )
                    propagate_grouped_tensorssa_info(node, grouped_tensors)
                    continue
            raise NotImplementedError(
                f"unsupported epilogue node: {node.format_node()}"
            )


def cute_epilogue_code(
    graph_module: torch.fx.GraphModule,
    *,
    fast_math: bool = False,
) -> tuple[
    list[str],
    str,
    list[torch.fx.Node],
    QuackLocalReduceInfo | None,
    QuackAuxOutputInfo | None,
    QuackMainOutputTransformInfo | None,
    tuple[str, ...],
]:
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
        ModificationWrapperCuteDSL,
    )
    from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
    from torch.utils._sympy.value_ranges import ValueRanges

    mm_node = find_single_gemm_node(graph_module)
    kernel = QuackCuteDSLKernel()
    handler = ModificationWrapperCuteDSL(kernel, 0, {}, None)
    placeholder_nodes = [
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    ]
    gemm_placeholder_nodes = {
        arg
        for arg in pytree.tree_leaves((mm_node.args, mm_node.kwargs))
        if isinstance(arg, torch.fx.Node)
    }
    aux_placeholder_nodes = [
        node
        for node in placeholder_nodes
        if node not in gemm_placeholder_nodes
        and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    env: dict[torch.fx.Node, Any] = {
        mm_node: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }
    for i, node in enumerate(aux_placeholder_nodes):
        env[node] = CuteDSLCSEVariable(
            f"aux{i}", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("QUACK GEMM epilogue backend expects one output")
    output_plan = analyze_output(unwrap_output(output_nodes[0]), mm_node)
    output_value = output_plan.output_value
    local_reduce = output_plan.local_reduce
    local_norm = output_plan.local_norm
    aux_output = output_plan.aux_output
    main_output_transform = output_plan.main_output_transform

    if main_output_transform is not None:
        if local_reduce is not None or aux_output is not None or aux_placeholder_nodes:
            raise NotImplementedError(
                "QUACK shape-changing main epilogues cannot be combined with aux outputs or captured tensor reads yet"
            )

    compile_pointwise_nodes(
        graph_module,
        mm_node,
        output_plan.skip_nodes,
        env,
        kernel,
        handler,
        fast_math=fast_math,
    )

    if local_norm is not None and local_norm.dim == 0:
        output_value = local_norm.source_node

    result = str(
        cute_arg(output_value, env)
        if isinstance(output_value, torch.fx.Node)
        else output_value
    )
    if aux_output is not None:
        result = f"({result}, {cute_arg(aux_output.output_value, env)})"
    elif local_reduce is not None and local_reduce.epilogue_reduce_source_node is not None:
        result = f"({result}, {cute_arg(local_reduce.epilogue_reduce_source_node, env)})"

    return (
        kernel.body.lines,
        result,
        aux_placeholder_nodes,
        local_reduce,
        aux_output,
        main_output_transform,
        main_output_transform.concat_layout if main_output_transform is not None else (),
    )


def render_gemm_epilogue(
    epilogue_name: str, body_lines: list[str], result: str, aux_args: list[str]
) -> str:
    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.kernel.mm_common import load_kernel_template

    template = KernelTemplate._template_from_string(
        load_kernel_template("quack_gemm_epilogue")
    )
    if template is None:
        raise ImportError("jinja2 is required to render QuACK GEMM epilogue templates")
    return template.render(
        epilogue_name=epilogue_name,
        body_lines=body_lines,
        result=result,
        aux_args=aux_args,
    )


def materialize_quack_epilogue(
    graph_module: torch.fx.GraphModule,
    *,
    fast_math: bool = False,
) -> tuple[
    str,
    str,
    list[torch.fx.Node],
    QuackLocalReduceInfo | None,
    QuackAuxOutputInfo | None,
    QuackMainOutputTransformInfo | None,
    tuple[str, ...],
]:
    (
        lines,
        result,
        aux_placeholder_nodes,
        local_reduce,
        aux_output,
        main_output_transform,
        concat_layout,
    ) = cute_epilogue_code(graph_module, fast_math=fast_math)
    key = (
        "flex_gemm_quack_epilogue_"
        + hashlib.sha256(f"fast_math={fast_math}\n{graph_module.code}".encode()).hexdigest()[:16]
    )
    return (
        key,
        render_gemm_epilogue(
            key, lines, result, [f"aux{i}" for i in range(len(aux_placeholder_nodes))]
        ),
        aux_placeholder_nodes,
        local_reduce,
        aux_output,
        main_output_transform,
        concat_layout,
    )
