# mypy: allow-untyped-defs
import hashlib
import inspect
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    torch.ops.aten._scaled_mm_v2.default: GemmEpilogueOpInfo(
        "scaled_mm_v2", 0, 1, supports_quack=False
    ),
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
_SUPPORTED_GEMM_OP_NAMES = "mm/addmm/bmm/baddbmm/_scaled_mm/_scaled_mm_v2/_grouped_mm"


@torch.library.custom_op("flex_gemm::mx_e8m0_scale", mutates_args=())
def mx_e8m0_scale(amax: torch.Tensor, max_power: int = 8) -> torch.Tensor:
    """Encode FLOOR-mode MX scale from an absolute max as biased E8M0.

    This small op captures the nontrivial MX scale policy without turning the
    whole quantization expression into a private marker. The returned dtype is
    the biased E8M0 representation; consumers can call ``.float()`` to recover
    the represented power-of-two scale for arithmetic.
    """
    mbits_f32 = 23
    f32_exp_bias = 127
    e8m0_exp_bias = 127
    e8m0_nan_val = 255
    max_abs = amax.to(torch.float32)
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((torch.bitwise_right_shift(max_abs_int32, mbits_f32)) & 0xFF) - f32_exp_bias
    scale_e8m0_unbiased = extracted_pow2 - max_power
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-e8m0_exp_bias, max=e8m0_exp_bias + 1
    )
    scale_e8m0_biased = (scale_e8m0_unbiased + e8m0_exp_bias).to(torch.uint8)
    scale_e8m0_biased = torch.where(
        torch.isnan(max_abs),
        torch.full_like(scale_e8m0_biased, e8m0_nan_val),
        scale_e8m0_biased,
    )
    return scale_e8m0_biased.view(torch.float8_e8m0fnu)


@mx_e8m0_scale.register_fake
def _(amax: torch.Tensor, max_power: int = 8) -> torch.Tensor:
    return torch.empty_strided(
        tuple(amax.shape),
        tuple(amax.stride()),
        device=amax.device,
        dtype=torch.float8_e8m0fnu,
    )


@torch.library.custom_op("flex_gemm::nvfp4_e4m3_scale", mutates_args=())
def nvfp4_e4m3_scale(amax: torch.Tensor) -> torch.Tensor:
    """Encode NVFP4 per-block scale as E4M3.

    This captures the NVFP4 block-scale policy: scale = clamp(amax / 6,
    tiny(e4m3), max(e4m3)).to(float8_e4m3fn).
    """
    scale = amax.to(torch.float32) / 6.0
    scale = torch.clamp(
        scale,
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=torch.finfo(torch.float8_e4m3fn).max,
    )
    return scale.to(torch.float8_e4m3fn)


@nvfp4_e4m3_scale.register_fake
def _(amax: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        tuple(amax.shape),
        tuple(amax.stride()),
        device=amax.device,
        dtype=torch.float8_e4m3fn,
    )


def _quack_f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    if x.dtype is not torch.float32:
        x = x.float()
    exp_bias = (1 << (ebits - 1)) - 1
    max_int = (1 << (ebits + mbits)) - 1
    sign_mask = 1 << (ebits + mbits)
    mbits_f32 = 23
    ebits_f32 = 8
    f32_exp_bias = 127
    magic_adder = (1 << (mbits_f32 - mbits - 1)) - 1
    max_normal = 2 ** ((1 << ebits) - 1 - exp_bias) * ((1 << (mbits + 1)) - 1) / (2**mbits)
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (f32_exp_bias - exp_bias) + (mbits_f32 - mbits) + 1
    denorm_mask_int = denorm_exp << mbits_f32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32, device=x.device).view(torch.float32)

    x_bits = x.view(torch.int32)
    sign = x_bits & 0x80000000
    abs_x = (x_bits ^ sign).view(torch.float32)
    saturate_mask = abs_x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), abs_x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = (abs_x + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = abs_x.view(torch.int32)
    mant_odd = (normal_x >> (mbits_f32 - mbits)) & 1
    val_to_add = ((exp_bias - f32_exp_bias) << mbits_f32) + magic_adder
    normal_x = ((normal_x + val_to_add + mant_odd) >> (mbits_f32 - mbits)).to(torch.uint8)

    out = torch.full_like(x, max_int, dtype=torch.uint8)
    out = torch.where(denormal_mask, denormal_x, out)
    out = torch.where(normal_mask, normal_x, out)
    sign_lp = ((sign >> (mbits_f32 + ebits_f32 - mbits - ebits)).to(torch.uint8)) & sign_mask
    return out | sign_lp


@torch.library.custom_op("flex_gemm::nvfp4_e2m1_pack", mutates_args=())
def nvfp4_e2m1_pack(x: torch.Tensor) -> torch.Tensor:
    """Pack logical E2M1 FP4 values into torch.float4_e2m1fn_x2 storage."""
    if x.shape[-1] % 2 != 0:
        raise RuntimeError("nvfp4_e2m1_pack requires an even last dimension")
    codes = _quack_f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    flat = codes.contiguous().view(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).view(*codes.shape[:-1], codes.shape[-1] // 2)
    return packed.view(torch.float4_e2m1fn_x2)


@nvfp4_e2m1_pack.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise RuntimeError("nvfp4_e2m1_pack requires an even last dimension")
    return torch.empty(
        (*tuple(x.shape[:-1]), x.shape[-1] // 2),
        device=x.device,
        dtype=torch.float4_e2m1fn_x2,
    )


def _normalize_gemm_epilogue_op(gemm_op: Callable[..., Any]) -> Callable[..., Any]:
    import torch.nn.functional as F

    return {
        **_GEMM_EPILOGUE_OP_ALIASES,
        F.grouped_mm: torch.ops.aten._grouped_mm.default,
    }.get(gemm_op, gemm_op)


def _as_list(value: Any | list[Any] | tuple[Any, ...] | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _enum_values(value: Any | list[Any] | tuple[Any, ...] | None) -> list[Any]:
    return [item.value if hasattr(item, "value") else item for item in _as_list(value)]


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


@dataclass(frozen=True)
class QuackLocalReduceInfo:
    view_node: torch.fx.Node
    reduce_node: torch.fx.Node
    source_node: torch.fx.Node
    keepdim: bool
    group_size: int
    dim: int
    feeds_main: bool = False
    kind: str = "sum"
    scale: float = 1.0
    max_power: int = 8
    output_node: torch.fx.Node | None = None
    extra_skip_nodes: frozenset[torch.fx.Node] = field(default_factory=frozenset)
    epilogue_reduce_source_node: torch.fx.Node | None = None
    epilogue_reduce_value_node: torch.fx.Node | None = None

    @property
    def aux_output_node(self) -> torch.fx.Node:
        return self.output_node if self.output_node is not None else self.reduce_node


@dataclass(frozen=True)
class QuackLocalNormInfo:
    output_node: torch.fx.Node
    div_node: torch.fx.Node
    view_node: torch.fx.Node
    reduce_node: torch.fx.Node
    source_node: torch.fx.Node
    group_size: int
    dim: int


@dataclass(frozen=True)
class QuackViewMatch:
    node: torch.fx.Node
    base: Any
    shape: Any


@dataclass(frozen=True)
class QuackSumMatch:
    node: torch.fx.Node
    view_node: Any
    dims: tuple[Any, ...]
    keepdim: Any
    dtype: Any


@dataclass(frozen=True)
class QuackAuxOutputInfo:
    output_value: torch.fx.Node
    group_size: int | None = None
    dim: int | None = None


@dataclass(frozen=True)
class QuackMainOutputTransformInfo:
    kind: str
    group_size: int | None = None


@dataclass(frozen=True)
class QuackOutputPlan:
    output_value: Any
    skip_nodes: frozenset[torch.fx.Node]
    local_reduce: QuackLocalReduceInfo | None
    local_norm: QuackLocalNormInfo | None
    aux_output: QuackAuxOutputInfo | None
    main_output_transform: QuackMainOutputTransformInfo | None = None


@dataclass(frozen=True)
class QuackGroupedTensorSSAInfo:
    group_size: int
    groups_per_fragment: int
    nonnegative: bool = False

    @property
    def keepdim_reshape(self) -> str:
        return f"((1, 1, {self.groups_per_fragment}), 1, 1)"


@dataclass(frozen=True)
class QuackTensorSSAReduceMatch:
    node: torch.fx.Node
    input_node: Any
    dims: tuple[Any, ...]
    keepdim: Any
    dtype: Any
    kind: str


@dataclass(frozen=True)
class QuackTensorSSAReduceDesc:
    cute_op: str
    init_val: str
    requires_nonnegative: bool = False


_QUACK_TENSORSSA_REDUCTIONS = {
    "sum": QuackTensorSSAReduceDesc("cute.ReductionOp.ADD", "0.0"),
    "amax": QuackTensorSSAReduceDesc(
        "cute.ReductionOp.MAX", "0.0", requires_nonnegative=True
    ),
}


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


def _normalize_reduce_dims(dim: Any) -> tuple[Any, ...]:
    if isinstance(dim, (list, tuple)):
        return tuple(dim)
    return (dim,)


def _normalize_quack_shape(shape: Any) -> Any:
    if isinstance(shape, torch.Size):
        return tuple(shape)
    return shape


def _match_quack_view_or_reshape(node: Any) -> QuackViewMatch | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        return QuackViewMatch(node=node, base=node.args[0], shape=node.args[1])
    if node.op == "call_method" and node.target in ("view", "reshape"):
        return QuackViewMatch(node=node, base=node.args[0], shape=node.args[1:])
    return None


def _match_quack_acc_source(base: Any, mm_node: torch.fx.Node) -> torch.fx.Node | None:
    if base is mm_node:
        return mm_node
    if (
        isinstance(base, torch.fx.Node)
        and base.op == "call_function"
        and base.target
        in (
            torch.ops.aten._to_copy.default,
            torch.ops.prims.convert_element_type.default,
        )
        and base.args[0] is mm_node
    ):
        return base
    return None


def _match_quack_sum(node: Any, *, allow_method_sum: bool) -> QuackSumMatch | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target == torch.ops.aten.sum.dim_IntList:
        view_node = node.args[0]
        dims = _normalize_reduce_dims(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        )
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return QuackSumMatch(
            node=node, view_node=view_node, dims=dims, keepdim=keepdim, dtype=dtype
        )
    if allow_method_sum and node.op == "call_method" and node.target == "sum":
        if len(node.args) < 2:
            return None
        view_node = node.args[0]
        dims = _normalize_reduce_dims(node.args[1])
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.kwargs.get("dtype")
        return QuackSumMatch(
            node=node, view_node=view_node, dims=dims, keepdim=keepdim, dtype=dtype
        )
    return None


def _match_quack_tensorssa_reduce(node: Any) -> QuackTensorSSAReduceMatch | None:
    sum_match = _match_quack_sum(node, allow_method_sum=True)
    if sum_match is not None:
        return QuackTensorSSAReduceMatch(
            node=sum_match.node,
            input_node=sum_match.view_node,
            dims=sum_match.dims,
            keepdim=sum_match.keepdim,
            dtype=sum_match.dtype,
            kind="sum",
        )
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amax.default,
        torch.amax,
    ):
        input_node = node.args[0]
        dims = _normalize_reduce_dims(
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        )
        keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return QuackTensorSSAReduceMatch(
            node=node,
            input_node=input_node,
            dims=dims,
            keepdim=keepdim,
            dtype=None,
            kind="amax",
        )
    if node.op == "call_method" and node.target == "amax":
        if len(node.args) < 2:
            return None
        return QuackTensorSSAReduceMatch(
            node=node,
            input_node=node.args[0],
            dims=_normalize_reduce_dims(node.args[1]),
            keepdim=node.kwargs.get("keepdim", False),
            dtype=None,
            kind="amax",
        )
    return None


def _match_quack_local_n_amax_reduce(
    node: Any, mm_node: torch.fx.Node, *, scale: float = 1.0
) -> QuackLocalReduceInfo | None:
    reduce_match = _match_quack_tensorssa_reduce(node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not isinstance(abs_node, torch.fx.Node):
        return None
    is_abs = (
        abs_node.op == "call_function"
        and abs_node.target in (torch.ops.aten.abs.default, torch.abs)
    ) or (abs_node.op == "call_method" and abs_node.target == "abs")
    if not is_abs:
        return None
    view_match = _match_quack_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = _match_quack_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = _quack_grouped_n_fragment_shape(_normalize_quack_shape(view_match.shape))
    if not _is_quack_same_fragment_n_group_shape(shape):
        return None
    reduce_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if reduce_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // shape[-1])
    if tuple(reduce_meta.shape) != tuple(expected_shape):
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_node=reduce_match.node,
        source_node=source_node,
        keepdim=False,
        group_size=shape[-1],
        dim=1,
        kind="amax_abs",
        scale=scale,
        extra_skip_nodes=frozenset((abs_node, node)),
    )


def _match_quack_scaled_local_n_amax_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op != "call_function":
        return None
    target = node.target
    args = node.args
    if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if isinstance(rhs, (int, float)):
            local_reduce = _match_quack_local_n_amax_reduce(
                lhs, mm_node, scale=float(rhs)
            )
        elif isinstance(lhs, (int, float)):
            local_reduce = _match_quack_local_n_amax_reduce(
                rhs, mm_node, scale=float(lhs)
            )
        else:
            return None
    elif target in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if not isinstance(rhs, (int, float)) or rhs == 0:
            return None
        local_reduce = _match_quack_local_n_amax_reduce(
            lhs, mm_node, scale=1.0 / float(rhs)
        )
    else:
        return None
    if local_reduce is None:
        return None
    return QuackLocalReduceInfo(
        view_node=local_reduce.view_node,
        reduce_node=local_reduce.reduce_node,
        source_node=local_reduce.source_node,
        keepdim=local_reduce.keepdim,
        group_size=local_reduce.group_size,
        dim=local_reduce.dim,
        feeds_main=local_reduce.feeds_main,
        kind=local_reduce.kind,
        scale=local_reduce.scale,
        max_power=local_reduce.max_power,
        output_node=node,
        extra_skip_nodes=local_reduce.extra_skip_nodes | frozenset((node,)),
    )


def _split_quack_scalar_scale(node: Any) -> tuple[Any, float] | None:
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    target = node.target
    args = node.args
    if target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, operator.mul):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if isinstance(rhs, (int, float)):
            return lhs, float(rhs)
        if isinstance(lhs, (int, float)):
            return rhs, float(lhs)
        return None
    if target in (
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        operator.truediv,
    ):
        if len(args) < 2:
            return None
        lhs, rhs = args[:2]
        if not isinstance(rhs, (int, float)) or rhs == 0:
            return None
        return lhs, 1.0 / float(rhs)
    return None


def _match_quack_local_n_amax_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    aux_output_node = node
    extra_skip_nodes: set[torch.fx.Node] = set()
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target
        in (
            torch.ops.aten._to_copy.default,
            torch.ops.prims.convert_element_type.default,
        )
    ):
        extra_skip_nodes.add(node)
        node = node.args[0]
    aux_view = _match_quack_view_or_reshape(node)
    if aux_view is None:
        return None
    extra_skip_nodes.add(aux_view.node)
    if isinstance(aux_view.base, torch.fx.Node):
        extra_skip_nodes.add(aux_view.base)
    scaled = _split_quack_scalar_scale(aux_view.base)
    if scaled is None:
        # Also accept the unscaled canonical keepdim form:
        #   scale = x.abs().amax(-1, keepdim=True)
        #   return main, scale.view(M, -1)
        reduce_node = aux_view.base
        scale = 1.0
    else:
        reduce_node, scale = scaled
    reduce_match = _match_quack_tensorssa_reduce(reduce_node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if not bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not isinstance(abs_node, torch.fx.Node):
        return None
    is_abs = (
        abs_node.op == "call_function"
        and abs_node.target in (torch.ops.aten.abs.default, torch.abs)
    ) or (abs_node.op == "call_method" and abs_node.target == "abs")
    if not is_abs:
        return None
    view_match = _match_quack_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = _match_quack_acc_source(view_match.base, mm_node)
    epilogue_reduce_source_node = None
    if source_node is None:
        if not _quack_output_uses_node(view_match.base, mm_node):
            return None
        source_node = view_match.base
        epilogue_reduce_source_node = view_match.node
    grouped_shape = _quack_grouped_n_fragment_shape(
        _normalize_quack_shape(view_match.shape)
    )
    if not _is_quack_same_fragment_n_group_shape(grouped_shape):
        return None
    aux_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if aux_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // grouped_shape[-1])
    if tuple(aux_meta.shape) != tuple(expected_shape):
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_node=reduce_node,
        source_node=source_node,
        keepdim=False,
        group_size=grouped_shape[-1],
        dim=1,
        kind="amax_abs",
        scale=scale,
        output_node=aux_output_node,
        extra_skip_nodes=frozenset(extra_skip_nodes | {reduce_node, abs_node}),
        epilogue_reduce_source_node=epilogue_reduce_source_node,
        epilogue_reduce_value_node=aux_view.base,
    )


def _match_quack_local_n_primitive_scale_view(
    node: Any,
    mm_node: torch.fx.Node,
    *,
    target: Any,
    kind: str,
    max_power: int = 8,
) -> QuackLocalReduceInfo | None:
    aux_view = _match_quack_view_or_reshape(node)
    if aux_view is None:
        return None
    scale_node = aux_view.base
    if not (
        isinstance(scale_node, torch.fx.Node)
        and scale_node.op == "call_function"
        and scale_node.target == target
    ):
        return None
    op_max_power = max_power
    if target == torch.ops.flex_gemm.mx_e8m0_scale.default:
        op_max_power = scale_node.args[1] if len(scale_node.args) > 1 else scale_node.kwargs.get("max_power", max_power)
        if not isinstance(op_max_power, int):
            return None
    reduce_node = scale_node.args[0]
    reduce_match = _match_quack_tensorssa_reduce(reduce_node)
    if reduce_match is None or reduce_match.kind != "amax":
        return None
    if reduce_match.dims not in ((-1,), (2,)) or reduce_match.dtype is not None:
        return None
    if not bool(reduce_match.keepdim):
        return None
    abs_node = reduce_match.input_node
    if not isinstance(abs_node, torch.fx.Node):
        return None
    is_abs = (
        abs_node.op == "call_function"
        and abs_node.target in (torch.ops.aten.abs.default, torch.abs)
    ) or (abs_node.op == "call_method" and abs_node.target == "abs")
    if not is_abs:
        return None
    view_match = _match_quack_view_or_reshape(abs_node.args[0])
    if view_match is None:
        return None
    source_node = _match_quack_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    grouped_shape = _quack_grouped_n_fragment_shape(
        _normalize_quack_shape(view_match.shape)
    )
    if not _is_quack_same_fragment_n_group_shape(grouped_shape):
        return None
    aux_meta = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    mm_meta = mm_node.meta.get("val")
    if aux_meta is None or mm_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // grouped_shape[-1])
    if tuple(aux_meta.shape) != tuple(expected_shape):
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_node=reduce_node,
        source_node=source_node,
        keepdim=False,
        group_size=grouped_shape[-1],
        dim=1,
        kind=kind,
        max_power=op_max_power,
        output_node=aux_view.node,
        extra_skip_nodes=frozenset((scale_node, reduce_node, abs_node, aux_view.node)),
    )


def _match_quack_local_n_mx_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    return _match_quack_local_n_primitive_scale_view(
        node,
        mm_node,
        target=torch.ops.flex_gemm.mx_e8m0_scale.default,
        kind="mx_e8m0_scale",
    )


def _match_quack_local_n_nvfp4_scale_view(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    return _match_quack_local_n_primitive_scale_view(
        node,
        mm_node,
        target=torch.ops.flex_gemm.nvfp4_e4m3_scale.default,
        kind="nvfp4_e4m3_scale",
    )


def _match_quack_local_n_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    sum_match = _match_quack_sum(node, allow_method_sum=True)
    if sum_match is None:
        return None
    if sum_match.dims not in ((-1,), (2,)) or sum_match.dtype is not None:
        return None
    view_match = _match_quack_view_or_reshape(sum_match.view_node)
    if view_match is None:
        return None
    source_node = _match_quack_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = _normalize_quack_shape(view_match.shape)
    group_shape = _quack_grouped_n_fragment_shape(shape)
    if not _is_quack_n_group_shape(group_shape):
        return None
    group_size = group_shape[-1]
    mm_meta = mm_node.meta.get("val")
    reduce_meta = sum_match.node.meta.get("val")
    if mm_meta is None or reduce_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (
        (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // group_size, 1)
        if bool(sum_match.keepdim)
        else (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // group_size)
    )
    if tuple(reduce_meta.shape) != expected_shape:
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_node=sum_match.node,
        source_node=source_node,
        keepdim=bool(sum_match.keepdim),
        group_size=group_size,
        dim=1,
    )


def _match_quack_local_m_reduce(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalReduceInfo | None:
    sum_match = _match_quack_sum(node, allow_method_sum=False)
    if sum_match is None:
        return None
    if sum_match.dims not in ((1,), (2,)) or sum_match.dtype is not None:
        return None
    view_match = _match_quack_view_or_reshape(sum_match.view_node)
    if view_match is None:
        return None
    source_node = _match_quack_acc_source(view_match.base, mm_node)
    if source_node is None:
        return None
    shape = _normalize_quack_shape(view_match.shape)
    if not isinstance(shape, (list, tuple)) or len(shape) not in (3, 4):
        return None
    group_dim = 1 if len(shape) == 3 else 2
    if sum_match.dims != (group_dim,):
        return None
    if shape[-3] != -1 or not isinstance(shape[-2], int) or shape[-2] <= 0:
        return None
    mm_val = mm_node.meta.get("val")
    reduce_val = sum_match.node.meta.get("val")
    if mm_val is None or reduce_val is None:
        return None
    mm_shape = tuple(mm_val.shape)
    if len(mm_shape) not in (2, 3) or shape[-1] != mm_shape[-1]:
        return None
    group_size = shape[-2]
    expected_shape = (
        (*mm_shape[:-2], mm_shape[-2] // group_size, 1, mm_shape[-1])
        if sum_match.keepdim
        else (*mm_shape[:-2], mm_shape[-2] // group_size, mm_shape[-1])
    )
    if tuple(reduce_val.shape) != expected_shape:
        return None
    return QuackLocalReduceInfo(
        view_node=view_match.node,
        reduce_node=sum_match.node,
        source_node=source_node,
        keepdim=bool(sum_match.keepdim),
        group_size=group_size,
        dim=0,
    )


def _match_quack_local_norm(
    node: Any,
    mm_node: torch.fx.Node,
    reduce_matcher: Callable[[Any, torch.fx.Node], QuackLocalReduceInfo | None],
    *,
    dim: int,
) -> QuackLocalNormInfo | None:
    output_view = _match_quack_view_or_reshape(node)
    if output_view is None:
        return None
    div_node = output_view.base
    output_shape = _normalize_quack_shape(output_view.shape)
    if not isinstance(div_node, torch.fx.Node):
        return None
    if not (div_node.op == "call_function" and div_node.target == torch.ops.aten.div.Tensor):
        return None
    lhs, rhs = div_node.args[:2]
    local_reduce = reduce_matcher(rhs, mm_node)
    if local_reduce is None or lhs is not local_reduce.view_node:
        return None
    output_meta = node.meta.get("val")
    mm_meta = mm_node.meta.get("val")
    if not isinstance(output_shape, (list, tuple)):
        return None
    if mm_meta is not None and tuple(output_shape) != tuple(mm_meta.shape):
        return None
    if output_meta is not None and mm_meta is not None:
        if tuple(output_meta.shape) != tuple(mm_meta.shape):
            return None
    if not local_reduce.keepdim:
        return None
    return QuackLocalNormInfo(
        output_node=node,
        div_node=div_node,
        view_node=local_reduce.view_node,
        reduce_node=local_reduce.reduce_node,
        source_node=local_reduce.source_node,
        group_size=local_reduce.group_size,
        dim=dim,
    )


def _match_quack_local_n_norm(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalNormInfo | None:
    return _match_quack_local_norm(
        node, mm_node, _match_quack_local_n_reduce, dim=1
    )


def _match_quack_local_m_norm(
    node: Any, mm_node: torch.fx.Node
) -> QuackLocalNormInfo | None:
    return _match_quack_local_norm(
        node, mm_node, _match_quack_local_m_reduce, dim=0
    )


def _quack_output_uses_node(value: Any, needle: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()

    def visit(item: Any) -> bool:
        if item is needle:
            return True
        if not isinstance(item, torch.fx.Node) or item in seen:
            return False
        seen.add(item)
        return any(visit(arg) for arg in pytree.tree_leaves((item.args, item.kwargs)))

    return visit(value)


def _match_quack_grouped_n_select(
    node: Any, mm_node: torch.fx.Node
) -> tuple[torch.fx.Node, int, int] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    if not (
        node.op == "call_function"
        and node.target == torch.ops.aten.select.int
        and len(node.args) >= 3
        and isinstance(node.args[2], int)
    ):
        return None
    view = _match_quack_view_or_reshape(node.args[0])
    view_shape = _normalize_quack_shape(view.shape) if view is not None else None
    if not (
        node.args[1] == -1
        or (
            isinstance(view_shape, (list, tuple))
            and node.args[1] == len(view_shape) - 1
        )
    ):
        return None
    if view is None or not _quack_output_uses_node(view.base, mm_node):
        return None
    shape = _quack_grouped_n_fragment_shape(view_shape)
    if not (
        _is_quack_same_fragment_n_group_shape(shape)
        and 0 <= node.args[2] < shape[-1]
    ):
        return None
    return view.node, node.args[2], shape[-1]


def _uses_quack_grouped_m_select(value: Any, mm_node: torch.fx.Node) -> bool:
    seen: set[torch.fx.Node] = set()

    def visit(node: Any) -> bool:
        if not isinstance(node, torch.fx.Node) or node in seen:
            return False
        seen.add(node)
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.select.int
            and len(node.args) >= 3
            and node.args[1] == 1
        ):
            view = _match_quack_view_or_reshape(node.args[0])
            shape = _normalize_quack_shape(view.shape) if view is not None else None
            if (
                view is not None
                and _quack_output_uses_node(view.base, mm_node)
                and isinstance(shape, (list, tuple))
                and len(shape) == 3
                and shape[0] == -1
                and isinstance(shape[1], int)
                and shape[1] > 0
            ):
                return True
        return any(visit(arg) for arg in pytree.tree_leaves((node.args, node.kwargs)))

    return visit(value)


def _match_quack_grouped_n_contract_main(
    output_value: Any, mm_node: torch.fx.Node
) -> QuackMainOutputTransformInfo | None:
    select_view = None
    select_group = None
    saw_select = False
    seen: set[torch.fx.Node] = set()

    def visit(value: Any) -> bool:
        nonlocal saw_select, select_view, select_group
        if not isinstance(value, torch.fx.Node):
            return True
        if value in seen:
            return True
        seen.add(value)
        select = _match_quack_grouped_n_select(value, mm_node)
        if select is not None:
            view_node, _index, group_size = select
            if select_view is None:
                select_view = view_node
                select_group = group_size
            elif select_view is not view_node or select_group != group_size:
                return False
            saw_select = True
            return True
        if value is mm_node:
            return False
        if (
            _match_quack_view_or_reshape(value) is not None
            and _quack_output_uses_node(value, mm_node)
        ):
            return False
        return all(visit(arg) for arg in pytree.tree_leaves((value.args, value.kwargs)))

    if not visit(output_value) or not saw_select:
        return None
    if select_group not in (2, 4):
        raise NotImplementedError(
            "QUACK grouped_n_contract currently supports only groups 2 and 4; "
            f"group={select_group} needs a validated epilogue store layout"
        )
    mm_meta = mm_node.meta.get("val")
    output_meta = (
        output_value.meta.get("val") if isinstance(output_value, torch.fx.Node) else None
    )
    if mm_meta is None or output_meta is None or len(mm_meta.shape) not in (2, 3):
        return None
    expected_shape = (*tuple(mm_meta.shape[:-1]), mm_meta.shape[-1] // select_group)
    if tuple(output_meta.shape) != expected_shape:
        return None
    return QuackMainOutputTransformInfo(
        kind="grouped_n_contract",
        group_size=select_group,
    )


def _analyze_quack_output(output_value: Any, mm_node: torch.fx.Node) -> QuackOutputPlan:
    local_reduce = None
    aux_output = None
    local_norm = _match_quack_local_n_norm(output_value, mm_node) or _match_quack_local_m_norm(
        output_value, mm_node
    )
    skip_nodes: set[torch.fx.Node] = set()
    if local_norm is not None and local_norm.dim == 0:
        skip_nodes.update(
            (
                local_norm.output_node,
                local_norm.div_node,
                local_norm.view_node,
                local_norm.reduce_node,
            )
        )
        local_reduce = QuackLocalReduceInfo(
            view_node=local_norm.view_node,
            reduce_node=local_norm.reduce_node,
            source_node=local_norm.source_node,
            keepdim=True,
            group_size=local_norm.group_size,
            dim=0,
            feeds_main=True,
        )
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        elif len(output_value) == 2:
            aux_value = output_value[1]
            local_reduce = (
                _match_quack_local_n_reduce(aux_value, mm_node)
                or _match_quack_local_m_reduce(aux_value, mm_node)
                or _match_quack_local_n_amax_reduce(aux_value, mm_node)
                or _match_quack_scaled_local_n_amax_reduce(aux_value, mm_node)
                or _match_quack_local_n_amax_scale_view(aux_value, mm_node)
                or _match_quack_local_n_mx_scale_view(aux_value, mm_node)
                or _match_quack_local_n_nvfp4_scale_view(aux_value, mm_node)
            )
            if local_reduce is not None and not local_reduce.keepdim:
                if (
                    local_reduce.epilogue_reduce_value_node is not None
                    and _quack_output_uses_node(
                        output_value[0], local_reduce.epilogue_reduce_value_node
                    )
                ):
                    local_reduce = replace(
                        local_reduce,
                        kind="copy",
                        scale=1.0,
                        epilogue_reduce_source_node=local_reduce.epilogue_reduce_value_node,
                    )
                if not _quack_output_uses_node(output_value[0], local_reduce.aux_output_node):
                    skip_nodes.add(local_reduce.aux_output_node)
                if not _quack_output_uses_node(output_value[0], local_reduce.reduce_node):
                    skip_nodes.add(local_reduce.reduce_node)
                for skip_node in local_reduce.extra_skip_nodes:
                    if not _quack_output_uses_node(output_value[0], skip_node):
                        skip_nodes.add(skip_node)
                if not _quack_output_uses_node(output_value[0], local_reduce.view_node):
                    skip_nodes.add(local_reduce.view_node)
                    if local_reduce.source_node is not mm_node:
                        skip_nodes.add(local_reduce.source_node)
            elif isinstance(aux_value, torch.fx.Node):
                aux_meta = aux_value.meta.get("val")
                mm_meta = mm_node.meta.get("val")
                if aux_meta is None or mm_meta is None:
                    raise NotImplementedError(
                        "QUACK generic aux tuple epilogues require fake tensor metadata"
                    )
                if tuple(aux_meta.shape) != tuple(mm_meta.shape):
                    raise NotImplementedError(
                        "QUACK generic aux tuple epilogues currently require the aux output "
                        "shape to match the GEMM output shape"
                    )
                aux_output = QuackAuxOutputInfo(output_value=aux_value)
            else:
                raise NotImplementedError(
                    "QUACK tuple epilogue currently supports only a supported local-reduce "
                    "aux output or one same-shape generic aux expression"
                )
            output_value = output_value[0]
        else:
            raise NotImplementedError(
                "QUACK GEMM epilogue backend expects one output or one supported "
                "local-reduce aux output"
            )
    main_output_transform = _match_quack_grouped_n_contract_main(output_value, mm_node)
    if main_output_transform is None and isinstance(output_value, torch.fx.Node):
        mm_meta = mm_node.meta.get("val")
        output_meta = output_value.meta.get("val")
        if (
            mm_meta is not None
            and output_meta is not None
            and tuple(output_meta.shape) != tuple(mm_meta.shape)
        ):
            if _uses_quack_grouped_m_select(output_value, mm_node):
                raise NotImplementedError(
                    "QUACK M-mode shape-changing main epilogues such as "
                    "acc.view(-1, group_m, N)[:, i, :] are not supported yet"
                )
            raise NotImplementedError(
                "QUACK shape-changing main epilogues currently require a supported "
                "local shape transform such as acc.view(M, -1, 2)[..., i]"
            )
    return QuackOutputPlan(
        output_value=output_value,
        skip_nodes=frozenset(skip_nodes),
        local_reduce=local_reduce,
        local_norm=local_norm,
        aux_output=aux_output,
        main_output_transform=main_output_transform,
    )


def _emit_quack_tensorssa_expr(
    kernel: _QuackCuteDSLKernel,
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


def _emit_quack_tensorssa_reshape(
    kernel: _QuackCuteDSLKernel, value: Any, shape: str
) -> Any:
    return _emit_quack_tensorssa_expr(kernel, f"{value}.reshape({shape})", like=value)


def _emit_quack_tensorssa_reduce(
    kernel: _QuackCuteDSLKernel,
    value: Any,
    *,
    op: str,
    init_val: str,
    reduction_profile: str,
) -> Any:
    return _emit_quack_tensorssa_expr(
        kernel,
        f"{value}.reduce({op}, init_val={init_val}, reduction_profile={reduction_profile})",
        like=value,
    )


def _emit_quack_tensorssa_broadcast_to(
    kernel: _QuackCuteDSLKernel, value: Any, shape: str
) -> Any:
    return _emit_quack_tensorssa_expr(kernel, f"{value}.broadcast_to({shape})", like=value)


def _quack_grouped_n_fragment_shape(shape: Any) -> Any:
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        return shape[-3:]
    return shape


def _is_quack_n_group_shape(shape: Any) -> bool:
    return (
        isinstance(shape, (list, tuple))
        and len(shape) == 3
        and shape[-2] == -1
        and isinstance(shape[-1], int)
        and shape[-1] > 0
    )


def _is_quack_same_fragment_n_group_shape(shape: Any) -> bool:
    return _is_quack_n_group_shape(shape) and 32 % shape[-1] == 0


def _lower_quack_view_or_reshape_node(
    node: torch.fx.Node,
    view_match: QuackViewMatch,
    env: dict[torch.fx.Node, Any],
    kernel: _QuackCuteDSLKernel,
    mm_node: torch.fx.Node,
) -> Any:
    source = _quack_cute_arg(view_match.base, env)
    shape = _normalize_quack_shape(view_match.shape)
    group_shape = _quack_grouped_n_fragment_shape(shape)
    if _is_quack_n_group_shape(group_shape):
        if not _is_quack_same_fragment_n_group_shape(group_shape):
            raise NotImplementedError(
                "QUACK reductions feeding the main output currently require a "
                "power-of-two group size that divides the same-fragment N width 32; "
                "aux reductions can use other static groups"
            )
        group_size = group_shape[-1]
        return _emit_quack_tensorssa_reshape(
            kernel, source, f"((1, {group_size}, {32 // group_size}), 1, 1)"
        )
    mm_meta = mm_node.meta.get("val")
    if (
        mm_meta is not None
        and isinstance(shape, (list, tuple))
        and tuple(shape) == tuple(mm_meta.shape)
    ):
        return _emit_quack_tensorssa_reshape(kernel, source, f"{env[mm_node]}.shape")
    raise NotImplementedError(f"unsupported QUACK epilogue view/reshape: {node.format_node()}")


def _lower_quack_grouped_n_select_node(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: _QuackCuteDSLKernel,
    grouped_tensors: dict[torch.fx.Node, QuackGroupedTensorSSAInfo],
) -> Any | None:
    if not (
        node.op == "call_function"
        and node.target == torch.ops.aten.select.int
        and len(node.args) >= 3
        and isinstance(node.args[2], int)
    ):
        return None
    view_node = node.args[0]
    view = _match_quack_view_or_reshape(view_node)
    view_shape = _normalize_quack_shape(view.shape) if view is not None else None
    if not (
        node.args[1] == -1
        or (
            isinstance(view_shape, (list, tuple))
            and node.args[1] == len(view_shape) - 1
        )
    ):
        return None
    info = grouped_tensors.get(view_node)
    if info is None or not (0 <= node.args[2] < info.group_size):
        return None
    source = _quack_cute_arg(view_node, env)
    return _emit_quack_tensorssa_expr(
        kernel,
        f"{source}[((0, {node.args[2]}, None), None, None)]",
        like=source,
    )


def _lower_quack_tensorssa_reduce_node(
    reduce_match: QuackTensorSSAReduceMatch,
    env: dict[torch.fx.Node, Any],
    kernel: _QuackCuteDSLKernel,
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
    desc = _QUACK_TENSORSSA_REDUCTIONS[reduce_match.kind]
    if desc.requires_nonnegative and not info.nonnegative:
        raise NotImplementedError(
            "QUACK amax TensorSSA reduction currently requires an abs/nonnegative input"
        )
    source = _quack_cute_arg(reduce_match.input_node, env)
    reduced = _emit_quack_tensorssa_reduce(
        kernel,
        source,
        op=desc.cute_op,
        init_val=desc.init_val,
        reduction_profile="((None, 1, None), 1, 1)",
    )
    if bool(reduce_match.keepdim):
        return _emit_quack_tensorssa_broadcast_to(
            kernel,
            _emit_quack_tensorssa_reshape(kernel, reduced, info.keepdim_reshape),
            f"{source}.shape",
        )
    return reduced


def _is_quack_nonnegative_expr(node: Any) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op == "call_function" and node.target in (
        torch.ops.aten.abs.default,
        torch.abs,
        torch.ops.aten.relu.default,
        torch.relu,
    ):
        return True
    if node.op == "call_method" and node.target in ("abs", "relu"):
        return True
    if node.op == "call_function" and node.target in (
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    ):
        return _is_quack_nonnegative_expr(node.args[0])
    if node.op == "call_function" and node.target == torch.ops.aten.clamp.default:
        min_value = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
        return isinstance(min_value, (int, float)) and min_value >= 0
    if node.op == "call_method" and node.target == "clamp":
        min_value = node.kwargs.get("min", None)
        return isinstance(min_value, (int, float)) and min_value >= 0
    return False


def _propagate_quack_grouped_tensorssa_info(
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


def _compile_quack_pointwise_nodes(
    graph_module: torch.fx.GraphModule,
    mm_node: torch.fx.Node,
    skip_nodes: frozenset[torch.fx.Node],
    env: dict[torch.fx.Node, Any],
    kernel: _QuackCuteDSLKernel,
    handler: Any,
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
                view_match = _match_quack_view_or_reshape(node)
                if view_match is not None:
                    shape = _normalize_quack_shape(view_match.shape)
                    env[node] = _lower_quack_view_or_reshape_node(
                        node, view_match, env, kernel, mm_node
                    )
                    group_shape = _quack_grouped_n_fragment_shape(shape)
                    if _is_quack_same_fragment_n_group_shape(group_shape):
                        base_info = grouped_tensors.get(view_match.base)
                        grouped_tensors[node] = QuackGroupedTensorSSAInfo(
                            group_size=group_shape[-1],
                            groups_per_fragment=32 // group_shape[-1],
                            nonnegative=(
                                base_info.nonnegative
                                if base_info is not None
                                else _is_quack_nonnegative_expr(view_match.base)
                            ),
                        )
                    continue
                select_value = _lower_quack_grouped_n_select_node(
                    node, env, kernel, grouped_tensors
                )
                if select_value is not None:
                    env[node] = select_value
                    continue
                reduce_match = _match_quack_tensorssa_reduce(node)
                if reduce_match is not None:
                    env[node] = _lower_quack_tensorssa_reduce_node(
                        reduce_match, env, kernel, grouped_tensors
                    )
                    continue
                if node.op == "call_method":
                    arg = _quack_cute_arg(node.args[0], env)
                    if node.target == "relu":
                        env[node] = ops.relu(arg).value
                        _propagate_quack_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == "abs":
                        input_info = grouped_tensors.get(node.args[0])
                        if input_info is not None and input_info.nonnegative:
                            env[node] = arg
                            grouped_tensors[node] = input_info
                        else:
                            env[node] = ops.abs(arg).value
                            _propagate_quack_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == "clamp":
                        result = arg
                        if node.kwargs.get("min") is not None:
                            result = ops.maximum(result, node.kwargs["min"]).value
                        if node.kwargs.get("max") is not None:
                            result = ops.minimum(result, node.kwargs["max"]).value
                        env[node] = result
                        _propagate_quack_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                if node.op == "call_function":
                    if node.target in (torch.ops.aten.abs.default, torch.abs):
                        input_info = grouped_tensors.get(node.args[0])
                        if input_info is not None and input_info.nonnegative:
                            env[node] = _quack_cute_arg(node.args[0], env)
                            grouped_tensors[node] = input_info
                            continue
                    if node.target == torch.ops.flex_gemm.mx_e8m0_scale.default:
                        source = _quack_cute_arg(node.args[0], env)
                        max_power = node.args[1] if len(node.args) > 1 else node.kwargs.get("max_power", 8)
                        scale_exp = f"(cute.math.floor(cute.math.log2({source})) - {max_power})"
                        env[node] = _emit_quack_tensorssa_expr(
                            kernel,
                            "cute.math.exp2("
                            f"cute.where({scale_exp} < -127.0, -127.0, "
                            f"cute.where({scale_exp} > 128.0, 128.0, {scale_exp}))"
                            ")",
                            like=source,
                        )
                        _propagate_quack_grouped_tensorssa_info(node, grouped_tensors)
                        continue
                    if node.target == torch.ops.flex_gemm.nvfp4_e4m3_scale.default:
                        raise NotImplementedError(
                            "QUACK nvfp4_e4m3_scale feeding the main output requires "
                            "E4M3 rounding semantics and is not implemented yet"
                        )
                    env[node] = _quack_cute_call_function(
                        node.target,
                        tuple(_quack_cute_arg(arg, env) for arg in node.args),
                        {
                            key: _quack_cute_arg(value, env)
                            for key, value in node.kwargs.items()
                        },
                    )
                    _propagate_quack_grouped_tensorssa_info(node, grouped_tensors)
                    continue
            raise NotImplementedError(
                f"unsupported epilogue node: {node.format_node()}"
            )


def _quack_cute_epilogue_code(
    graph_module: torch.fx.GraphModule,
) -> tuple[
    list[str],
    str,
    list[torch.fx.Node],
    QuackLocalReduceInfo | None,
    QuackAuxOutputInfo | None,
    QuackMainOutputTransformInfo | None,
]:
    from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
        ModificationWrapperCuteDSL,
    )
    from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
    from torch.utils._sympy.value_ranges import ValueRanges

    mm_node = _find_single_quack_gemm_node(graph_module)
    kernel = _QuackCuteDSLKernel()
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
    output_plan = _analyze_quack_output(_unwrap_output(output_nodes[0]), mm_node)
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

    _compile_quack_pointwise_nodes(
        graph_module, mm_node, output_plan.skip_nodes, env, kernel, handler
    )

    if local_norm is not None and local_norm.dim == 0:
        output_value = local_norm.source_node

    result = str(
        _quack_cute_arg(output_value, env)
        if isinstance(output_value, torch.fx.Node)
        else output_value
    )
    if aux_output is not None:
        result = f"({result}, {_quack_cute_arg(aux_output.output_value, env)})"
    elif local_reduce is not None and local_reduce.epilogue_reduce_source_node is not None:
        result = f"({result}, {_quack_cute_arg(local_reduce.epilogue_reduce_source_node, env)})"

    return (
        kernel.body.lines,
        result,
        aux_placeholder_nodes,
        local_reduce,
        aux_output,
        main_output_transform,
    )


def _render_quack_gemm_epilogue(
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
) -> tuple[
    str,
    str,
    list[torch.fx.Node],
    QuackLocalReduceInfo | None,
    QuackAuxOutputInfo | None,
    QuackMainOutputTransformInfo | None,
]:
    lines, result, aux_placeholder_nodes, local_reduce, aux_output, main_output_transform = (
        _quack_cute_epilogue_code(graph_module)
    )
    key = (
        "flex_gemm_quack_epilogue_"
        + hashlib.sha256(graph_module.code.encode()).hexdigest()[:16]
    )
    return (
        key,
        _render_quack_gemm_epilogue(
            key, lines, result, [f"aux{i}" for i in range(len(aux_placeholder_nodes))]
        ),
        aux_placeholder_nodes,
        local_reduce,
        aux_output,
        main_output_transform,
    )


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

    if gemm_op == torch.ops.aten._scaled_mm_v2.default:
        if len(gemm_args) != 4:
            raise RuntimeError(
                "_scaled_mm_v2 epilogue fusion expects gemm_args=(mat_a, mat_b, scale_a, scale_b)"
            )
        scale_a_args = _as_list(gemm_args[2])
        scale_b_args = _as_list(gemm_args[3])
        scale_a_len = len(scale_a_args)
        body_kwargs = dict(gemm_kwargs)
        recipe_a = _enum_values(
            body_kwargs.pop("recipe_a", body_kwargs.pop("scale_recipe_a", None))
        )
        recipe_b = _enum_values(
            body_kwargs.pop("recipe_b", body_kwargs.pop("scale_recipe_b", None))
        )
        swizzle_a = _enum_values(body_kwargs.pop("swizzle_a", None))
        swizzle_b = _enum_values(body_kwargs.pop("swizzle_b", None))
        bias = body_kwargs.pop("bias", None)
        out_dtype = body_kwargs.pop(
            "out_dtype", body_kwargs.pop("output_dtype", torch.bfloat16)
        )
        contraction_dim = list(body_kwargs.pop("contraction_dim", ()))
        use_fast_accum = body_kwargs.pop("use_fast_accum", False)
        if body_kwargs:
            raise RuntimeError(f"unsupported _scaled_mm_v2 kwargs: {body_kwargs}")

        def body_fn(mat_a, mat_b, *scale_args):
            return epilogue_fn(
                torch.ops.aten._scaled_mm_v2.default(
                    mat_a,
                    mat_b,
                    list(scale_args[:scale_a_len]),
                    recipe_a,
                    swizzle_a,
                    list(scale_args[scale_a_len:]),
                    recipe_b,
                    swizzle_b,
                    bias,
                    out_dtype,
                    contraction_dim,
                    use_fast_accum,
                )
            )

        return _gemm_epilogue_fusion(
            gemm_op,
            body_fn,
            (gemm_args[0], gemm_args[1], *scale_a_args, *scale_b_args),
            {},
            kernel_options,
        )

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
