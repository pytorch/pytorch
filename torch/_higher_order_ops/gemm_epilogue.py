# mypy: allow-untyped-defs
import inspect
import math
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
_SUPPORTED_BACKENDS = {"TRITON", "CUTEDSL", "QUACK"}
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


@torch.library.custom_op("flex_gemm::silu_tanh", mutates_args=())
def silu_tanh(x: torch.Tensor) -> torch.Tensor:
    """SiLU through the tanh identity for QUACK fastmath epilogue lowering.

    This is numerically the same expression used by QuACK's optimized built-in
    ``activation=\"silu-tanh\"`` path:

        silu(x) = h * tanh(h) + h, where h = 0.5 * x

    Eager uses PyTorch tanh. QUACK TensorSSA lowering uses
    ``cute.math.tanh(..., fastmath=True)`` so the generated code uses the fast
    MUFU.TANH path instead of generic exp/rcp SiLU lowering.
    """
    h = x * 0.5
    return h * torch.tanh(h) + h


@silu_tanh.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )


@torch.library.custom_op("flex_gemm::tanh_fast", mutates_args=())
def tanh_fast(x: torch.Tensor) -> torch.Tensor:
    """Tanh marker for QUACK fastmath epilogue lowering.

    Eager uses ``torch.tanh``. QUACK lowers this op to
    ``cute.math.tanh(..., fastmath=True)`` to select the compact MUFU.TANH path
    for activation-style approximate tanh epilogues.
    """
    return torch.tanh(x)


@tanh_fast.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
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
        if any(
            isinstance(value, torch.Tensor) for value in pytree.tree_leaves(gemm_kwargs)
        ):
            raise RuntimeError(
                "gemm_kwargs must not contain tensor values; pass tensor inputs through gemm_args"
            )

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


def _pop_kwarg_alias(
    kwargs: dict[str, Any], primary: str, alias: str, default: Any = None
) -> Any:
    has_primary = primary in kwargs
    has_alias = alias in kwargs
    if has_primary and has_alias:
        raise RuntimeError(f"cannot specify both {primary!r} and {alias!r}")
    if has_primary:
        return kwargs.pop(primary)
    if has_alias:
        return kwargs.pop(alias)
    return default


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
            _pop_kwarg_alias(body_kwargs, "recipe_a", "scale_recipe_a")
        )
        recipe_b = _enum_values(
            _pop_kwarg_alias(body_kwargs, "recipe_b", "scale_recipe_b")
        )
        swizzle_a = _enum_values(body_kwargs.pop("swizzle_a", None))
        swizzle_b = _enum_values(body_kwargs.pop("swizzle_b", None))
        bias = body_kwargs.pop("bias", None)
        out_dtype = _pop_kwarg_alias(
            body_kwargs, "out_dtype", "output_dtype", torch.bfloat16
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

    kernel_options = (
        {"backend": "TRITON"} if kernel_options is None else dict(kernel_options)
    )
    kernel_options["grouped_mm_has_offs"] = offs is not None
    kernel_options["grouped_mm_has_bias"] = bias is not None
    return _gemm_epilogue_fusion(
        torch.ops.aten._grouped_mm.default,
        body_fn,
        tuple(gemm_args),
        {},
        kernel_options,
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
