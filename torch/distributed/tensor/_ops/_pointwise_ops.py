# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import infer_broadcast_dims_map
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate


aten = torch.ops.aten
prims = torch.ops.prims

# Linear pointwise ops, split by linearity type.
unary_linear_ops = [
    aten.to.device,
    aten.to.dtype,
    aten.to.prim_Device,
    aten.to.prim_dtype,
    aten.to.prim_other,
]


def _common_pointwise_single_dim_strategy(
    partial_extra_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> Callable[
    [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
]:
    """Factory for single-dim strategies that add partial placement rules.

    Returns strategies shaped [output, *args] only.  Tensor kwarg placements
    (e.g. ``out``, ``lr``) are appended by the wrapper in
    ``_register_single_dim_pointwise``.
    """

    def strategy(
        op: OpOverload,
        args_schema: ArgsType,
        kwargs_schema: KwargsType,
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        tensor_arg_metas: list[TensorMeta] = [
            arg for arg in args_schema if isinstance(arg, TensorMeta)
        ]
        common_shape = torch.broadcast_shapes(
            *[arg.shape for arg in args_schema if isinstance(arg, TensorMeta)]
        )
        # For multi-output ops (e.g. frexp), all outputs share the same
        # pointwise sharding, so replicate the output placement.
        num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
        placements: list[list[Placement | _ShardingPlaceholder]] = []
        for i in range(len(common_shape)):
            shard_placements: list[Placement | _ShardingPlaceholder] = [
                _ShardingPlaceholder(i)
            ] * num_outputs
            for arg in tensor_arg_metas:
                common_dim_to_arg_dim = infer_broadcast_dims_map(
                    common_shape, arg.shape
                )
                # If the output shard dim maps to an input dim, shard that
                # input dim; otherwise it was broadcast, so replicate.
                if common_dim_to_arg_dim[i] >= 0:
                    shard_placements.append(
                        _ShardingPlaceholder(common_dim_to_arg_dim[i])
                    )
                else:
                    shard_placements.append(Replicate())
            placements.append(shard_placements)
        if partial_extra_rules:
            n_tensors = len(tensor_arg_metas)
            expected_len = num_outputs + n_tensors
            for rule in partial_extra_rules:
                # Filter rather than assert: some ops (e.g. mul.Tensor) mix
                # unary rules (len 2, for scalar promotion) and binary rules
                # (len 3, for tensor-tensor), so mismatched lengths are expected.
                # see _MUL_RULES to see how _UNARY_LINEAR_RULES handles the
                # scalar promotion case
                if len(rule) == expected_len:
                    placements.append(rule)
        return placements

    return strategy


def _is_list_op(op: OpOverload) -> bool:
    """Returns True if op is a foreach, amp_foreach, or fused op."""
    name = op.name()
    return name.startswith(("aten::_foreach_", "aten::_amp_foreach_", "aten::_fused_"))


# The state_steps arg of fused adam / adamw is a Replicate scalar tensor, which will be put on
# the compute_mesh of an op across all parameter groups, even when not all parameter groups
# are on the same device mesh. This idx will help avoid hitting exceptions or unnecessary
# redistribute during sharding propagation.
_FUSED_OP_SCALAR_IDX = 5

# Ops registered with extra Partial rules; populated by _register_single_dim_pointwise
# when partial_extra_rules is not None, to avoid double-registration from tag discovery.
_specially_registered_ops: set[OpOverload] = set()


def _register_single_dim_pointwise(
    op: OpOverload,
    partial_extra_rules: list[list[Placement]] | None = None,
    static_argnum: int = 0,
) -> None:
    if partial_extra_rules is not None:
        _specially_registered_ops.add(op)
    inner_fn = _common_pointwise_single_dim_strategy(
        partial_extra_rules=partial_extra_rules  # pyrefly: ignore[bad-argument-type]
    )

    # Wrap to append tensor kwarg placements in schema declaration order.
    # out = output placement (s[0]); everything else (e.g. lr) = Replicate.
    # TODO: move kwargs handling upstream if this works
    def strategy_fn(
        op: OpOverload,
        args: ArgsType,
        kwargs: KwargsType,
        _fn: Callable = inner_fn,
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        strategies = _fn(op, args, kwargs)
        kw_names = [k for k, v in kwargs.items() if isinstance(v, TensorMeta)]
        if not kw_names:
            return strategies
        return [
            s + [s[0] if name == "out" else Replicate() for name in kw_names]
            for s in strategies
        ]

    if _is_list_op(op):
        schema_info = RuntimeSchemaInfo(needs_pytree=True)
    else:
        schema_info = RuntimeSchemaInfo(static_argnum, static_kwargkey=["out"])
    # Fused ops (e.g. _fused_adam_) have state_steps on a potentially different
    # mesh; see the note in expand_to_full_mesh_op_strategy for details.
    different_mesh_args: list[int] | None = None
    if op.name().startswith("aten::_fused_"):
        different_mesh_args = [_FUSED_OP_SCALAR_IDX]
    register_single_dim_strategy(
        op,
        schema_info=schema_info,
        allow_uneven_sharding=True,
        allow_unbacked_sharding=True,
        different_mesh_args=different_mesh_args,
    )(strategy_fn)


_UNARY_LINEAR_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg")],
]

binary_additive_ops = [
    aten.add.Tensor,
    aten.add_.Tensor,
    aten.add.out,
    aten.sub.Tensor,
    aten.sub_.Tensor,
    aten.sub.out,
    # foreach variants
    aten._foreach_add.List,
    aten._foreach_add_.List,
    aten._foreach_sub.List,
    aten._foreach_sub_.List,
]

_BINARY_ADDITIVE_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg"), Partial("avg")],
    # P(x), R -> P(x): adding/subtracting a replicated value preserves partial types
    # avg, max, min. sum would result in R being added n times, n = num_ranks
    # (the replicated value is constant across ranks, so reduce order is unaffected)
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("min"), Partial("min"), Replicate()],
    # R, P(avg) -> P(avg): avg is linear so this holds for any alpha
    # (R, P(max/min) excluded: negative alpha would flip the ordering)
    [Partial("avg"), Replicate(), Partial("avg")],
]

for op in binary_additive_ops:
    _register_single_dim_pointwise(op, _BINARY_ADDITIVE_RULES)

# mul: partials propagate through either arg. div: only through numerator.
binary_mul_ops = [
    aten.mul.Tensor,
    aten.mul_.Tensor,
    aten.mul.out,
    # foreach variants
    aten._foreach_mul.List,
    aten._foreach_mul_.List,
    aten._foreach_mul.Tensor,
    aten._foreach_mul_.Tensor,
]
binary_div_ops = [
    aten.div.Tensor,
    aten.div_.Tensor,
    aten.div.out,
    # foreach variants
    aten._foreach_div.List,
    aten._foreach_div_.List,
    aten._foreach_div.Tensor,
    aten._foreach_div_.Tensor,
]

# _UNARY_LINEAR_RULES handles the scalar promotion case: Python's __mul__/__truediv__
# promote scalars to 0-dim tensors, so aten.mul.Scalar dispatches as aten.mul.Tensor
# with n_tensors=1, matching the length-2 unary rules.
_MUL_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("sum"), Replicate(), Partial("sum")],
    [Partial("avg"), Replicate(), Partial("avg")],
]

_DIV_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
]

for op in binary_mul_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES + _MUL_RULES)

for op in binary_div_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES + _DIV_RULES)

scalar_linear_ops = [
    aten.div.Scalar,
    aten.div_.Scalar,
    aten.mul.Scalar,
    aten.mul_.Scalar,
    # foreach variants
    aten._foreach_div.Scalar,
    aten._foreach_div_.Scalar,
    aten._foreach_mul.Scalar,
    aten._foreach_mul_.Scalar,
    aten._foreach_div.ScalarList,
    aten._foreach_div_.ScalarList,
    aten._foreach_mul.ScalarList,
    aten._foreach_mul_.ScalarList,
]

for op in scalar_linear_ops:
    _register_single_dim_pointwise(op, _UNARY_LINEAR_RULES, static_argnum=1)


# Non-decreasing unary ops: f(max(a,b)) = max(f(a),f(b)).
# Only ops that are non-decreasing on their ENTIRE domain belong here.
# Ops with restricted domains (e.g. log on (0,∞), asin on [-1,1]) do NOT qualify
# because P(max) offsets can push inputs outside the valid domain.
non_decreasing_unary_ops = [
    aten.asinh.default,
    aten.asinh_.default,
    aten.asinh.out,
    aten.atan.default,
    aten.atan_.default,
    aten.atan.out,
    aten.ceil.default,
    aten.ceil_.default,
    aten.ceil.out,
    aten.deg2rad.default,
    aten.deg2rad_.default,
    aten.deg2rad.out,
    aten.erf.default,
    aten.erf_.default,
    aten.erf.out,
    aten.exp.default,
    aten.exp_.default,
    aten.exp.out,
    aten.exp2.default,
    aten.exp2_.default,
    aten.exp2.out,
    aten.expm1.default,
    aten.expm1_.default,
    aten.expm1.out,
    aten.floor.default,
    aten.floor_.default,
    aten.floor.out,
    aten.rad2deg.default,
    aten.rad2deg_.default,
    aten.rad2deg.out,
    aten.relu.default,
    aten.relu_.default,
    aten.round.decimals,
    aten.round.default,
    aten.round_.decimals,
    aten.round_.default,
    aten.round.decimals_out,
    aten.round.out,
    aten.sgn.default,
    aten.sgn_.default,
    aten.sgn.out,
    aten.sigmoid.default,
    aten.sigmoid_.default,
    aten.sigmoid.out,
    aten.sign.default,
    aten.sign_.default,
    aten.sign.out,
    aten.sinh.default,
    aten.sinh_.default,
    aten.sinh.out,
    aten.tanh.default,
    aten.tanh_.default,
    aten.tanh.out,
    aten.trunc.default,
    aten.trunc_.default,
    aten.trunc.out,
    # nan_to_num is non-decreasing on its entire domain (including nan/inf):
    # it maps -inf→min, nan→0, inf→max, and is identity elsewhere.
    aten.nan_to_num.default,
    aten.nan_to_num_.default,
    aten.nan_to_num.out,
    # hardshrink: x if |x|>lambd else 0. Non-decreasing on entire domain.
    aten.hardshrink.default,
    # I1(x) is monotonically non-decreasing for all real x.
    aten.special_modified_bessel_i1.default,
    # threshold(x, t, v): x if x > t else v. Non-decreasing for v <= t (the
    # common case, including the default v=0, t=0).
    aten.threshold.default,
    # foreach variants
    aten._foreach_exp.default,
    aten._foreach_exp_.default,
    aten._foreach_clamp_max_.Scalar,
    aten._foreach_clamp_min_.Scalar,
]

_NON_DECREASING_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("max")],
    [Partial("min"), Partial("min")],
]

for op in non_decreasing_unary_ops:
    _register_single_dim_pointwise(op, _NON_DECREASING_RULES)

# Non-increasing unary ops: f(max(a,b)) = min(f(a),f(b)).
# Note: acos excluded due to domain constraints [-1,1] causing validation failures
non_increasing_unary_ops: list[OpOverload] = [
    aten.erfc.default,
    aten.erfc_.default,
    aten.erfc.out,
    aten.special_erfcx.default,
    aten.special_erfcx.out,
]

_NON_INCREASING_RULES: list[list[Placement]] = [
    [Partial("min"), Partial("max")],
    [Partial("max"), Partial("min")],
]

for op in non_increasing_unary_ops:
    _register_single_dim_pointwise(op, _NON_INCREASING_RULES)

# Bessel K functions are strictly decreasing for x > 0 but undefined at x <= 0.
# Only P(min)->P(max) is safe: P(min) offsets add positive values to the
# non-holding rank, keeping all inputs positive. P(max) offsets subtract,
# which can push inputs to x <= 0 producing NaN.
_POSITIVE_DOMAIN_NON_INCREASING_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("min")],
]

for op in [
    aten.special_modified_bessel_k0.default,
    aten.special_modified_bessel_k1.default,
    aten.special_scaled_modified_bessel_k0.default,
    aten.special_scaled_modified_bessel_k1.default,
]:
    _register_single_dim_pointwise(op, _POSITIVE_DOMAIN_NON_INCREASING_RULES)

# neg is linear: -(A1 + A2) = -A1 + -A2
neg_ops = [
    aten.neg.default,
    aten.neg_.default,
    aten.neg.out,
    # foreach variants
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
]

_NEG_RULES: list[list[Placement]] = _UNARY_LINEAR_RULES + _NON_INCREASING_RULES

for op in neg_ops:
    _register_single_dim_pointwise(op, _NEG_RULES)

# xlog1py(x, y) = x * log1p(y). Linear in x with y replicated:
# (a+b)*log1p(y) = a*log1p(y) + b*log1p(y).
_XLOG1PY_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
]

for op in [aten.special_xlog1py.default, aten.special_xlog1py.other_scalar]:
    _register_single_dim_pointwise(op, _XLOG1PY_RULES)


# All-partial-preserving unary ops: P(x)->P(x) for all x.
# TODO: positive should be removed once CIA (Copy Is All) optimizes it away.
all_partial_preserving_unary_ops = [
    aten.to.device,
    aten.to.dtype,
    aten.to.prim_Device,
    aten.to.prim_dtype,
    aten.to.prim_other,
    aten.positive.default,
]

_ALL_PARTIAL_PRESERVING_RULES: list[list[Placement]] = [
    [Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")
]

for op in all_partial_preserving_unary_ops:
    _register_single_dim_pointwise(op, _ALL_PARTIAL_PRESERVING_RULES)

all_partial_preserving_binary_ops = [
    aten.copy_.default,
    aten.copy_.Tensor,
    prims.copy_to.default,
]

_ALL_PARTIAL_BINARY_PRESERVING_RULES: list[list[Placement]] = [
    [Partial(r), Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")
]

for op in all_partial_preserving_binary_ops:
    _register_single_dim_pointwise(op, _ALL_PARTIAL_BINARY_PRESERVING_RULES)

# Monotonic increasing in both args but don't preserve any specific partial type.
monotonic_binary_ops = [
    aten.logaddexp.default,
    aten.logaddexp.out,
    aten.logaddexp2.default,
    aten.logaddexp2.out,
]

_MONOTONE_BINARY_BASE_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("max"), Replicate(), Partial("max")],
    [Partial("min"), Partial("min"), Replicate()],
    [Partial("min"), Replicate(), Partial("min")],
]

for op in monotonic_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_BINARY_BASE_RULES)

# Binary ops monotonically increasing in both arguments.
# max-preserving: P(max)+P(max)->P(max) because max(max(a),max(b)) = max(a,b)
monotonic_max_preserving_binary_ops = [
    aten.clamp_min.Tensor,
    aten.fmax.default,
    aten.fmax.out,
    aten.maximum.default,
    aten.maximum.out,
    prims.fmax.default,
    # foreach variants
    aten._foreach_maximum_.List,
]

_MONOTONE_MAX_PRESERVING_BINARY_BASE_RULES: list[list[Placement]] = [
    *_MONOTONE_BINARY_BASE_RULES,
    [Partial("max"), Partial("max"), Partial("max")],
]

for op in monotonic_max_preserving_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_MAX_PRESERVING_BINARY_BASE_RULES)

# min-preserving: P(min)+P(min)->P(min) because min(min(a),min(b)) = min(a,b)
monotonic_min_preserving_binary_ops = [
    aten.clamp_max.Tensor,
    aten.fmin.default,
    aten.fmin.out,
    aten.minimum.default,
    aten.minimum.out,
    prims.fmin.default,
]

_MONOTONE_MIN_PRESERVING_BINARY_BASE_RULES: list[list[Placement]] = [
    *_MONOTONE_BINARY_BASE_RULES,
    [Partial("min"), Partial("min"), Partial("min")],
]

for op in monotonic_min_preserving_binary_ops:
    _register_single_dim_pointwise(op, _MONOTONE_MIN_PRESERVING_BINARY_BASE_RULES)


# Ops that are pointwise for DTensor purposes but lack torch.Tag.pointwise.
# TODO(pianpwk): add torch.Tag.pointwise to these ops in native_functions.yaml
# so this list can be removed.
_extra_pointwise_ops: list[OpOverload] = [
    aten.__irshift__.Scalar,
    aten.__irshift__.Tensor,
    aten._conj.default,
    aten._conj_physical.default,
    aten.abs.default,
    aten.abs.out,
    aten.abs_.default,
    aten.acos.default,
    aten.acos.out,
    aten.acos_.default,
    aten.acosh.default,
    aten.acosh.out,
    aten.acosh_.default,
    aten.add.Scalar,
    aten.add_.Scalar,
    aten.addcdiv.default,
    aten.addcdiv.out,
    aten.addcdiv_.default,
    aten.addcmul.default,
    aten.addcmul.out,
    aten.addcmul_.default,
    aten.angle.default,
    aten.angle.out,
    aten.asin.default,
    aten.asin.out,
    aten.asin_.default,
    aten.atan2.default,
    aten.atan2.out,
    aten.atan2_.default,
    aten.atanh.default,
    aten.atanh.out,
    aten.atanh_.default,
    aten.bitwise_and.Scalar,
    aten.bitwise_and.Scalar_Tensor,
    aten.bitwise_and.Scalar_out,
    aten.bitwise_and.Tensor,
    aten.bitwise_and.Tensor_out,
    aten.bitwise_and_.Scalar,
    aten.bitwise_and_.Tensor,
    aten.bitwise_left_shift.Scalar_Tensor,
    aten.bitwise_left_shift.Tensor,
    aten.bitwise_left_shift.Tensor_Scalar,
    aten.bitwise_left_shift.Tensor_Scalar_out,
    aten.bitwise_left_shift.Tensor_out,
    aten.bitwise_left_shift_.Tensor,
    aten.bitwise_left_shift_.Tensor_Scalar,
    aten.bitwise_not.default,
    aten.bitwise_not.out,
    aten.bitwise_not_.default,
    aten.bitwise_or.Scalar,
    aten.bitwise_or.Scalar_Tensor,
    aten.bitwise_or.Scalar_out,
    aten.bitwise_or.Tensor,
    aten.bitwise_or.Tensor_out,
    aten.bitwise_or_.Scalar,
    aten.bitwise_or_.Tensor,
    aten.bitwise_right_shift.Scalar_Tensor,
    aten.bitwise_right_shift.Tensor,
    aten.bitwise_right_shift.Tensor_Scalar,
    aten.bitwise_right_shift.Tensor_Scalar_out,
    aten.bitwise_right_shift.Tensor_out,
    aten.bitwise_right_shift_.Tensor,
    aten.bitwise_right_shift_.Tensor_Scalar,
    aten.bitwise_xor.Scalar,
    aten.bitwise_xor.Scalar_Tensor,
    aten.bitwise_xor.Scalar_out,
    aten.bitwise_xor.Tensor,
    aten.bitwise_xor.Tensor_out,
    aten.bitwise_xor_.Scalar,
    aten.bitwise_xor_.Tensor,
    aten.clamp.default,
    aten.clamp.Tensor,
    aten.clamp.out,
    aten.clamp_.default,
    aten.clamp_.Tensor,
    aten.clamp_min.default,
    aten.clamp_max.default,
    aten.clip.default,
    aten.clip.Tensor,
    aten.clip.out,
    aten.clip_.default,
    aten.clip_.Tensor,
    aten.conj_physical.default,
    aten.conj_physical.out,
    aten.conj_physical_.default,
    aten.complex.default,
    aten.complex.out,
    aten.copysign.Scalar,
    aten.copysign.Scalar_out,
    aten.copysign.Tensor,
    aten.copysign.out,
    aten.copysign_.Scalar,
    aten.copysign_.Tensor,
    aten.cos.default,
    aten.cos.out,
    aten.cos_.default,
    aten.cosh.default,
    aten.cosh.out,
    aten.cosh_.default,
    aten.digamma.default,
    aten.digamma.out,
    aten.digamma_.default,
    aten.div.Scalar_mode,
    aten.div.Tensor_mode,
    aten.div.out_mode,
    aten.div_.Scalar_mode,
    aten.div_.Tensor_mode,
    aten.eq.Tensor,
    aten.eq.Tensor_out,
    aten.eq.Scalar,
    aten.eq.Scalar_out,
    aten.erfinv.default,
    aten.erfinv.out,
    aten.erfinv_.default,
    aten.exponential_.default,
    aten.float_power.Scalar,
    aten.float_power.Scalar_out,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Tensor_Scalar_out,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Tensor_out,
    aten.float_power_.Scalar,
    aten.float_power_.Tensor,
    aten.fill_.Tensor,
    aten.fmod.Scalar,
    aten.fmod.Scalar_out,
    aten.fmod.Tensor,
    aten.fmod.Tensor_out,
    aten.fmod_.Scalar,
    aten.fmod_.Tensor,
    aten.frac.default,
    aten.frac.out,
    aten.frac_.default,
    aten.gcd.default,
    aten.gcd.out,
    aten.ge.Scalar,
    aten.ge.Tensor,
    aten.gelu.default,
    aten.gt.Tensor,
    aten.gt.Tensor_out,
    aten.gt.Scalar,
    aten.gt.Scalar_out,
    aten.gt.Scalar,
    aten.gt.Tensor,
    aten.hardshrink.default,
    aten.heaviside.default,
    aten.heaviside.out,
    aten.hypot.default,
    aten.hypot.out,
    aten.hypot_.default,
    aten.i0.default,
    aten.i0.out,
    aten.i0_.default,
    aten.igamma.default,
    aten.igamma.out,
    aten.igamma_.default,
    aten.igammac.default,
    aten.igammac.out,
    aten.igammac_.default,
    aten.isinf.default,
    aten.isnan.default,
    aten.isneginf.default,
    aten.isneginf.out,
    aten.isposinf.default,
    aten.isposinf.out,
    aten.ldexp.Tensor,
    aten.ldexp.out,
    aten.ldexp_.default,
    aten.lt.Tensor,
    aten.lt.Tensor_out,
    aten.lt.Scalar,
    aten.lt.Scalar_out,
    aten.le.Scalar,
    aten.le.Tensor,
    aten.lerp.Scalar,
    aten.lerp.Scalar_out,
    aten.lerp.Tensor,
    aten.lerp.Tensor_out,
    aten.lerp_.Scalar,
    aten.lerp_.Tensor,
    aten.lgamma.default,
    aten.lgamma.out,
    aten.lgamma_.default,
    aten.log.default,
    aten.log.out,
    aten.log10.default,
    aten.log10.out,
    aten.log10_.default,
    aten.log1p.default,
    aten.log1p.out,
    aten.log1p_.default,
    aten.log2.default,
    aten.log2.out,
    aten.log2_.default,
    aten.log_.default,
    aten.logical_and.default,
    aten.logical_and.out,
    aten.logical_and_.default,
    aten.logical_not.default,
    aten.logical_not.out,
    aten.logical_not_.default,
    aten.logical_or.default,
    aten.logical_or.out,
    aten.logical_or_.default,
    aten.logical_xor.default,
    aten.logical_xor.out,
    aten.logical_xor_.default,
    aten.logit.default,
    aten.logit.out,
    aten.logit_.default,
    aten.masked_fill.Scalar,
    aten.masked_fill.Tensor,
    aten.masked_fill_.Scalar,
    aten.masked_fill_.Tensor,
    aten.mvlgamma.default,
    aten.mvlgamma.out,
    aten.mvlgamma_.default,
    aten.native_dropout_backward.default,
    aten.native_dropout_backward.out,
    aten.ne.Scalar,
    aten.ne.Tensor,
    aten.nextafter.default,
    aten.nextafter.out,
    aten.nextafter_.default,
    aten.polygamma.default,
    aten.polygamma.out,
    aten.polygamma_.default,
    aten.pow.Scalar,
    aten.pow.Scalar_out,
    aten.pow.Tensor_Scalar,
    aten.pow.Tensor_Scalar_out,
    aten.pow.Tensor_Tensor,
    aten.pow.Tensor_Tensor_out,
    aten.polar.default,
    aten.pow_.Scalar,
    aten.pow_.Tensor,
    aten.reciprocal.default,
    aten.reciprocal.out,
    aten.reciprocal_.default,
    aten.remainder.Scalar,
    aten.remainder.Scalar_Tensor,
    aten.remainder.Scalar_out,
    aten.remainder.Tensor,
    aten.remainder.Tensor_out,
    aten.remainder_.Scalar,
    aten.remainder_.Tensor,
    aten.rsqrt.default,
    aten.rsqrt.out,
    aten.rsqrt_.default,
    aten.rrelu_with_noise.default,
    aten.rsub.Scalar,
    aten.rsub.Tensor,
    aten.signbit.default,
    aten.signbit.out,
    aten.silu.default,
    aten.silu.out,
    aten.sin.default,
    aten.sin.out,
    aten.sin_.default,
    aten.sinc.default,
    aten.sinc.out,
    aten.sinc_.default,
    aten.sqrt.default,
    aten.sqrt.out,
    aten.sqrt_.default,
    aten.square.default,
    aten.square.out,
    aten.square_.default,
    aten.sub.Scalar,
    aten.sub_.Scalar,
    aten.tan.default,
    aten.tan.out,
    aten.tan_.default,
    aten.threshold.default,
    aten.to.other,
    aten.true_divide.Scalar,
    aten.true_divide.Tensor,
    aten.where.Scalar,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
    aten.where.self,
    aten.where.self_out,
    aten.xlogy_.Scalar_Other,
    prims.bessel_i0e.default,
    prims.bessel_i1.default,
    prims.bessel_i1e.default,
    prims.bessel_j0.default,
    prims.bessel_j1.default,
    prims.div.default,
    prims.erfcx.default,
    prims.frexp.default,
    prims.gcd.default,
    prims.ndtri.default,
    prims.ne.default,
    prims.spherical_bessel_j0.default,
    prims.zeta.default,
    # special ops (pointwise, native C++ only — no Python decomposition)
    # please keep the entries below alphabetically sorted
    aten.special_airy_ai.default,
    aten.special_bessel_j0.default,
    aten.special_bessel_j1.default,
    aten.special_bessel_y0.default,
    aten.special_bessel_y1.default,
    aten.special_chebyshev_polynomial_t.default,
    aten.special_chebyshev_polynomial_t.n_scalar,
    aten.special_chebyshev_polynomial_t.x_scalar,
    aten.special_chebyshev_polynomial_u.default,
    aten.special_chebyshev_polynomial_u.n_scalar,
    aten.special_chebyshev_polynomial_u.x_scalar,
    aten.special_chebyshev_polynomial_v.default,
    aten.special_chebyshev_polynomial_v.n_scalar,
    aten.special_chebyshev_polynomial_v.x_scalar,
    aten.special_chebyshev_polynomial_w.default,
    aten.special_chebyshev_polynomial_w.n_scalar,
    aten.special_chebyshev_polynomial_w.x_scalar,
    aten.special_entr.default,
    aten.special_hermite_polynomial_h.default,
    aten.special_hermite_polynomial_h.n_scalar,
    aten.special_hermite_polynomial_h.x_scalar,
    aten.special_hermite_polynomial_he.default,
    aten.special_hermite_polynomial_he.n_scalar,
    aten.special_hermite_polynomial_he.x_scalar,
    aten.special_i0e.default,
    aten.special_i1.default,
    aten.special_i1e.default,
    aten.special_laguerre_polynomial_l.default,
    aten.special_laguerre_polynomial_l.n_scalar,
    aten.special_laguerre_polynomial_l.x_scalar,
    aten.special_legendre_polynomial_p.default,
    aten.special_legendre_polynomial_p.n_scalar,
    aten.special_legendre_polynomial_p.x_scalar,
    aten.special_log_ndtr.default,
    aten.special_modified_bessel_i0.default,
    aten.special_modified_bessel_i1.default,
    aten.special_modified_bessel_k0.default,
    aten.special_modified_bessel_k1.default,
    aten.special_ndtr.default,
    aten.special_ndtri.default,
    aten.special_polygamma.default,
    aten.special_scaled_modified_bessel_k0.default,
    aten.special_scaled_modified_bessel_k1.default,
    aten.special_shifted_chebyshev_polynomial_t.default,
    aten.special_shifted_chebyshev_polynomial_t.n_scalar,
    aten.special_shifted_chebyshev_polynomial_t.x_scalar,
    aten.special_shifted_chebyshev_polynomial_u.default,
    aten.special_shifted_chebyshev_polynomial_u.n_scalar,
    aten.special_shifted_chebyshev_polynomial_u.x_scalar,
    aten.special_shifted_chebyshev_polynomial_v.default,
    aten.special_shifted_chebyshev_polynomial_v.n_scalar,
    aten.special_shifted_chebyshev_polynomial_v.x_scalar,
    aten.special_shifted_chebyshev_polynomial_w.default,
    aten.special_shifted_chebyshev_polynomial_w.n_scalar,
    aten.special_shifted_chebyshev_polynomial_w.x_scalar,
    aten.special_spherical_bessel_j0.default,
    aten.special_xlog1py.default,
    aten.special_xlog1py.other_scalar,
    aten.special_xlog1py.self_scalar,
    aten.special_zeta.default,
    aten.special_zeta.other_scalar,
    aten.special_zeta.self_scalar,
    # foreach variants
    aten._foreach_abs.default,
    aten._foreach_abs_.default,
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv_.ScalarList,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul_.ScalarList,
    aten._foreach_addcmul_.Tensor,
    aten._foreach_lerp_.Scalar,
    aten._foreach_pow.List,
    aten._foreach_pow.ScalarList,
    aten._foreach_reciprocal_.default,
    aten._foreach_sub.Scalar,
    aten._foreach_sub_.Scalar,
    aten._foreach_sub.ScalarList,
    aten._foreach_sub_.ScalarList,
    aten._foreach_sqrt.default,
    aten._foreach_sqrt_.default,
    aten._foreach_zero_.default,
    aten._foreach_cos.default,
    aten._foreach_cos_.default,
    aten._foreach_log.default,
    aten._foreach_log_.default,
    aten._amp_foreach_non_finite_check_and_unscale_.default,
    # foreach linearity variants
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    # fused optimizer ops
    aten._fused_adam_.default,
    aten._fused_adam.default,
    aten._fused_adam.tensor_lr,
    aten._fused_adam_.tensor_lr,
    aten._fused_adamw_.default,
    aten._fused_adamw.default,
    aten._fused_adamw.tensor_lr,
    aten._fused_adamw_.tensor_lr,
]


def _get_pointwise_ops_from_tag() -> list[OpOverload]:
    """
    Auto-discover pointwise ops via torch.Tag.pointwise, from ops.aten, ops.prims.
    """
    ops = []
    for ns in [torch.ops.aten, torch.ops.prims]:
        for attr_name in dir(ns):
            attr = getattr(ns, attr_name)
            if isinstance(attr, torch._ops.OpOverloadPacket):
                for overload_name in attr.overloads():
                    op = getattr(attr, overload_name)
                    if torch.Tag.pointwise in op.tags:
                        ops.append(op)
    return ops


pointwise_ops = [
    op
    for op in _get_pointwise_ops_from_tag() + _extra_pointwise_ops
    if op not in _specially_registered_ops
]


for op in pointwise_ops:
    _register_single_dim_pointwise(op)


def register_inductor_prims() -> None:
    """Register DTensor sharding strategies for inductor prims ops.

    Called lazily because inductor prims are created via make_prim() in
    torch._inductor.inductor_prims, which is imported after this module.
    """
    # TODO: handle other inductor prims ops that may need DTensor sharding
    # strategies (e.g. mul_rn, div_rn). Those are more complicated and not
    # necessarily pointwise.
    _register_single_dim_pointwise(prims.fma.default)
