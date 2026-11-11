# Owner(s): ["module: onnx"]
"""Preserve pre-promoted inputs for the legacy ONNX Script adapter only."""

from __future__ import annotations

import onnx_ir as ir

import torch
from torch import _prims_common as prims
from torch.onnx._internal.exporter._torchlib.ops import _promotion
from torch.onnx._internal.fx import type_utils
from torch.utils import _pytree


# Captured from the former generated and curated rules, sharing in-place aliases.
_RULES = {
    name: (kind, positions)
    for kind, positions, names in (
        (
            "ALWAYS_BOOL",
            (0,),
            "isfinite isinf isnan isneginf isposinf isreal logical_not signbit",
        ),
        ("ALWAYS_BOOL", (0, 1), "eq ge gt le logical_and logical_or logical_xor lt ne"),
        ("BOOL_TO_LONG", (0,), "square"),
        ("COMPLEX_TO_FLOAT", (0,), "abs"),
        ("COMPLEX_TO_FLOAT", (0, 1), "l1_loss mse_loss smooth_l1_loss"),
        (
            "DEFAULT",
            (0,),
            "bitwise_not cauchy ceil celu elu exponential floor frac gelu geometric glu hardtanh leaky_relu log_normal mish neg nll_loss pdist relu round selu sgn sign softplus threshold trunc",
        ),
        (
            "DEFAULT",
            (0, 1),
            "add bitwise_and bitwise_left_shift bitwise_or bitwise_right_shift bitwise_xor dot floor_divide fmax fmin fmod gcd heaviside huber_loss hypot lcm logaddexp logaddexp2 maximum minimum mul prelu remainder sub vdot clamp_min clamp_max",
        ),
        ("DEFAULT", (0, 1, 2), "addcmul addr clamp lerp"),
        (
            "INT_TO_FLOAT",
            (0,),
            "acos acosh asin asinh atan atanh cos cosh deg2rad digamma erf erfc erfinv exp exp2 expm1 i0 lgamma log log10 log1p log2 logit logsumexp rad2deg reciprocal rsqrt sigmoid sin sinc sinh sqrt tan tanh",
        ),
        (
            "INT_TO_FLOAT",
            (0, 1),
            "atan2 copysign igamma igammac poisson_nll_loss true_divide xlogy",
        ),
        ("INT_TO_FLOAT", (0, 1, 2), "addcdiv"),
        ("NO_OPMATH", (0,), "cat fill"),
        ("NO_OPMATH", (0, 1), "nextafter"),
        ("NO_OPMATH", (1, 2), "where"),
        (
            "REDUCTION",
            (0,),
            "all any amax amin std std_mean var sum prod cumsum cumprod",
        ),
    )
    for name in names.split()
}


def promote_inputs(op, target, args, kwargs):
    """Apply the historical ATen policy without changing or executing FX nodes."""
    if not isinstance(target, torch._ops.OpOverload) or target.namespace != "aten":
        return args, False
    name = target.overloadpacket.__name__.removesuffix("_")
    if name == "div":
        kind = "INT_TO_FLOAT" if kwargs.get("rounding_mode") is None else "DEFAULT"
        rule = kind, (0, 1)
    else:
        rule = _RULES.get(name)
    if rule is None:
        return args, False
    kind, positions = rule
    positions = [i for i in positions if i < len(args) and args[i] is not None]
    if kind == "REDUCTION":
        restore = True
        if name in ("all", "any"):
            dtype = torch.bool
        else:
            dtype = kwargs.get("dtype", -1)
            _, result = _promotion.reduction_dtypes(
                op,
                args[0],
                dtype,
                integral_to_long=name in ("sum", "prod", "cumsum", "cumprod"),
            )
            dtype = prims.get_computation_dtype(_promotion._DTYPES[result])
    else:
        computation, result = _promotion.dtypes(
            op, *_pytree.tree_leaves([args[i] for i in positions]), kind=kind
        )
        dtype = result if kind in ("DEFAULT", "INT_TO_FLOAT") else computation
        restore = (
            kind in ("COMPLEX_TO_FLOAT", "BOOL_TO_LONG")
            and prims.get_computation_dtype(result) != result
        )
    promoted = list(args)
    for i in positions:
        promoted[i] = _pytree.tree_map(
            lambda value: value
            if type_utils.from_scalar_type_to_torch_dtype(type(value)) == dtype
            else _promotion.cast(op, value, dtype),
            args[i],
        )
    return promoted, restore


def restore_outputs(op, outputs, expected):
    if isinstance(outputs, ir.Value) and isinstance(expected, torch.Tensor):
        dtype = expected.dtype
        if dtype.is_complex:
            dtype = prims.corresponding_real_dtype(dtype)
        onnx_dtype = _promotion._ONNX_DTYPES[dtype]
        if outputs.dtype != onnx_dtype:
            outputs.shape = None
            return op.builder.node(
                "Cast",
                [outputs],
                attributes={"to": onnx_dtype},
                output_types=[ir.TensorType(onnx_dtype)],
            )[0]
    return outputs
