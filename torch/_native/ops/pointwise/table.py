# The pointwise op definition table (POINTWISE_DEF_TABLE): one PointwiseDef row per
# aten elementwise op. Each row is fully declarative -- the generic registration
# machinery (overrides.py) turns it into a (cond, impl) override, so adding an op is
# one row plus its kernel function in ops.py, not a hand-written override.
#
# This module is deliberately cutlass-FREE: it holds only the metadata registration
# needs (aten name, arity, promotion kind, scalar args, output-dtype policy) so that
# `import torch` -> override registration can read the table without pulling in the
# DSL runtime (the lazy-DSL-import contract; see test_no_dsl_imports_after_import_torch).
# The actual kernel math lives in ops.py, where every `fn` is a @cute.jit function;
# a row references it BY NAME (`fn`, a str) and overrides.py resolves the callable
# lazily via ops.get_fn(name) on the first real (non-declined) call.
#
# The named ops.py function is @cute.jit-able over COMPUTE-dtype scalars:
#   fn(*input_vals, *scalar_consts) -> result
# Inputs arrive already converted to the compute dtype; baked scalar args (e.g. add's
# `alpha`) follow, as compute-dtype constants. The result is cast to the op's output
# dtype. `fn` references only DSL ops (operators, cute.math.*), never a user-class
# method (which would trip the IR flattener).

from __future__ import annotations

import math
from typing import NamedTuple, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind


if TYPE_CHECKING:
    from collections.abc import Callable


class PointwiseDef(NamedTuple):
    aten: str  # aten op symbol incl overload, e.g. "add.Tensor", "neg"
    nin: int  # number of tensor inputs
    fn: str  # name of the @cute.jit kernel function in ops.py (resolved lazily)
    # aten elementwise type-promotion kind (single value, not combinable: a closed
    # Enum keying torch's elementwise_dtypes algorithm, not a bitwise Flag).
    promotion: PromotionKind = PromotionKind.DEFAULT
    scalars: tuple[str, ...] = ()  # positional arg names baked as compute consts
    # aten default for each scalar (aligned with `scalars`), used when the caller omits
    # it. () -> every default is 1 (add/sub alpha, addcmul value, elu's three). Ops with
    # non-1 aten defaults MUST set this (softplus threshold=20, *shrink lambd=0.5,
    # hardtanh [-1, 1]) or a defaulted call computes with the wrong constant.
    scalar_defaults: tuple = ()
    # Restrict the INPUT dtypes this override serves; inputs outside the set fall
    # back to aten. None -> the family default (all supported floats). Use to narrow
    # an op whose kernel is only correct for some dtypes (e.g. copysign excludes fp16).
    dtypes: tuple | None = None
    # Skip the .out variant because an ATEN kernel redispatches a WRAPPED-NUMBER tensor
    # into it. aten's pow_Scalar_out builds wrapped_scalar_tensor(base, exp.device()) --
    # a CUDA tensor flagged is_wrapped_number -- and calls at::pow_out (Pow.cpp:66). Our
    # router is a Python kernel, and converting a boxed wrapped-number tensor to Python
    # asserts it is CPU (pybind_utils.cpp toPyObject), so registering the .out overload
    # turns `2.0 ** cuda_tensor` / pow(2.0, t) / float_power(2.0, t) / integer ldexp into
    # an INTERNAL ASSERT. The assert fires in the boxed->Python conversion, BEFORE any
    # cond runs, so it cannot be declined in Python -- the overload must not be
    # registered at all. The functional variant still is.
    skip_out_variant: bool = False
    # EXTRA input precondition from this op's aten meta function, beyond dtype membership
    # and broadcasting: `(operands, target_or_None) -> bool`, where False means aten would
    # RAISE, so we must decline and let it. None -> no extra precondition. Two rows need
    # one, both because they REQUIRE a shared dtype rather than promoting:
    #   heaviside -- every operand and the out= tensor must share self's dtype
    #                ("heaviside is not yet implemented for tensors with different
    #                dtypes", BinaryOps.cpp TORCH_META_FUNC);
    #   lerp.Tensor -- `end` must match self always, and `weight` unless it is 0-dim,
    #                which promotes instead (Lerp.cpp TORCH_META_FUNC).
    # Serving a mismatch returns a value where aten errors, which is the divergence
    # test_heaviside and test_lerp_weight_tensor_promotion_error catch.
    precondition: Callable | None = None
    # OPTIONAL scalars: name -> fill-in for the entries of `scalars` that aten declares
    # `Scalar?`/`float?` and the caller may omit INDEPENDENTLY (clamp's min/max,
    # nan_to_num's nan/posinf/neginf, logit's eps). Distinct from scalar_defaults, which
    # is a fixed value per slot: these fill-ins depend on the RESULT DTYPE, so each is a
    # callable `(torch_dtype) -> value`. Verified against aten: an omitted nan_to_num
    # posinf is numeric_limits<dtype>::max() (fp16 65504, fp32 3.4e38, fp64 1.8e308), an
    # omitted clamp bound is the corresponding infinity, and an omitted logit eps is a
    # negative sentinel meaning "do not clamp" -- so one formula covers every arity.
    # Membership in this dict IS the "is optional" flag; there is no separate name list.
    optional_defaults: dict | None = None
    # Select the ops.py fn by a STRING kwarg.
    # aten spells two distinct kernels as one overload plus a mode string: gelu's
    # `approximate` in {"none","tanh"} and div's `rounding_mode` (None only here).
    # Maps kwarg VALUE -> fn name; the kwarg's own name is `mode_kwarg`.
    # A value absent from the map declines (aten validates it and raises).
    mode_kwarg: str | None = None
    mode_fns: dict | None = None


_INT2FLOAT = PromotionKind.INT_TO_FLOAT
_BOOL = PromotionKind.ALWAYS_BOOL


def _lowest(dt):
    # An omitted clamp lower bound is -inf for these floating inputs.
    return -math.inf


def _highest(dt):
    return math.inf


# --- `precondition` predicates (see PointwiseDef.precondition) ---


def _all_same_dtype(ins, tgt):
    # heaviside: every operand, and the out= tensor if there is one, must share self's
    # dtype -- aten does not promote here, it raises.
    return all(t.dtype == ins[0].dtype for t in ins) and (
        tgt is None or tgt.dtype == ins[0].dtype
    )


def _lerp_dtypes_ok(ins, tgt):
    # lerp.Tensor: `end` must match self; `weight` must too UNLESS it is 0-dim, which
    # aten promotes instead (promote_weight = weight.dim() == 0).
    self_, end, weight = ins
    return self_.dtype == end.dtype and (
        weight.dim() == 0 or weight.dtype == self_.dtype
    )


class PointwiseVariant(NamedTuple):
    # Functional calls allocate their result; .out calls keep the caller's output.
    name: str
    overload_suffix: str
    out_from: str


POINTWISE_VARIANTS: tuple[PointwiseVariant, ...] = (
    PointwiseVariant("functional", "", "alloc"),
    PointwiseVariant("out", "out", "out_kw"),
)

POINTWISE_DEF_TABLE: tuple[PointwiseDef, ...] = (
    PointwiseDef("neg", 1, "_neg"),
    PointwiseDef("add.Tensor", 2, "_add", scalars=("alpha",)),
    PointwiseDef("sub.Tensor", 2, "_sub", scalars=("alpha",)),
    PointwiseDef("mul.Tensor", 2, "_mul"),
    PointwiseDef(
        "div.Tensor",
        2,
        "_div",
        promotion=_INT2FLOAT,
    ),
    PointwiseDef("maximum", 2, "_maximum"),
    PointwiseDef("minimum", 2, "_minimum"),
    PointwiseDef("atan2", 2, "_atan2", promotion=_INT2FLOAT),
    PointwiseDef("rsub.Tensor", 2, "_rsub", scalars=("alpha",)),
    # fmax/fmin are the NaN-SUPPRESSING pair (vs maximum/minimum which propagate).
    PointwiseDef("fmax", 2, "_fmax"),
    PointwiseDef("fmin", 2, "_fmin"),
    PointwiseDef("clamp_min.Tensor", 2, "_clamp_min"),
    PointwiseDef("clamp_max.Tensor", 2, "_clamp_max"),
    PointwiseDef(
        "copysign.Tensor",
        2,
        "_copysign",
        promotion=_INT2FLOAT,
        # NO float16. copysign reads the sign BIT, and compute here is fp32, but widening
        # an fp16 NaN to fp32 CANONICALISES it and drops that bit (measured: -NaN fp16
        # widens to 0x7FFFFFFF, a positive NaN -- and ATen's own .float() does the same, so
        # its fp16 kernel must not go through fp32). bf16 is safe because bf16 -> fp32 is a
        # shift, which is why only fp16 is excluded. test_copysign_nan_sign covers this.
        dtypes=(torch.bfloat16, torch.float32),
    ),
    PointwiseDef("hypot", 2, "_hypot"),
    PointwiseDef("logaddexp", 2, "_logaddexp"),
    PointwiseDef("xlogy.Tensor", 2, "_xlogy", promotion=_INT2FLOAT),
    # pow keeps integers integer (aten int^int -> int); the float fn is the exp2/log2
    # composite with full sign/edge handling. Integer pow needs a repeated-multiply
    # loop the DSL can't express over a runtime exponent -> float-only.
    PointwiseDef("pow.Tensor_Tensor", 2, "_pow", skip_out_variant=True),
    # --- rounding / sign / activation (DEFAULT) ---
    # floor/ceil/trunc are no-ops on integers in aten AND cute.math.floor rejects Int, so
    # they stay float-only (an int input declines -> aten no-ops it). sign/relu are correct
    # on Int compute.
    PointwiseDef("floor", 1, "_floor"),
    PointwiseDef("ceil", 1, "_ceil"),
    PointwiseDef("trunc", 1, "_trunc"),
    PointwiseDef("sign", 1, "_sign"),
    PointwiseDef("relu", 1, "_relu"),
    PointwiseDef("abs", 1, "_abs"),
    PointwiseDef("square", 1, "_square"),
    # round/frac: no-op / undefined on integers like floor (round(int)=int in aten);
    # float-only for the same cute.math.floor reason.
    PointwiseDef("round", 1, "_round"),
    PointwiseDef("frac", 1, "_frac"),
    PointwiseDef("deg2rad", 1, "_deg2rad", promotion=_INT2FLOAT),
    PointwiseDef("rad2deg", 1, "_rad2deg", promotion=_INT2FLOAT),
    # --- unary transcendental math (INT_TO_FLOAT: int input -> float output) ---
    PointwiseDef("exp", 1, "_exp", promotion=_INT2FLOAT),
    PointwiseDef("exp2", 1, "_exp2", promotion=_INT2FLOAT),
    PointwiseDef("expm1", 1, "_expm1", promotion=_INT2FLOAT),
    PointwiseDef("log", 1, "_log", promotion=_INT2FLOAT),
    PointwiseDef("log2", 1, "_log2", promotion=_INT2FLOAT),
    PointwiseDef("log10", 1, "_log10", promotion=_INT2FLOAT),
    PointwiseDef("log1p", 1, "_log1p", promotion=_INT2FLOAT),
    PointwiseDef("sqrt", 1, "_sqrt", promotion=_INT2FLOAT),
    PointwiseDef("rsqrt", 1, "_rsqrt", promotion=_INT2FLOAT),
    PointwiseDef("reciprocal", 1, "_reciprocal", promotion=_INT2FLOAT),
    PointwiseDef("sin", 1, "_sin", promotion=_INT2FLOAT),
    PointwiseDef("cos", 1, "_cos", promotion=_INT2FLOAT),
    PointwiseDef("tan", 1, "_tan", promotion=_INT2FLOAT),
    PointwiseDef("asin", 1, "_asin", promotion=_INT2FLOAT),
    PointwiseDef("acos", 1, "_acos", promotion=_INT2FLOAT),
    PointwiseDef("atan", 1, "_atan", promotion=_INT2FLOAT),
    PointwiseDef("tanh", 1, "_tanh", promotion=_INT2FLOAT),
    PointwiseDef("erf", 1, "_erf", promotion=_INT2FLOAT),
    PointwiseDef("erfc", 1, "_erfc", promotion=_INT2FLOAT),
    PointwiseDef("sigmoid", 1, "_sigmoid", promotion=_INT2FLOAT),
    PointwiseDef("cosh", 1, "_cosh", promotion=_INT2FLOAT),
    PointwiseDef("sinh", 1, "_sinh", promotion=_INT2FLOAT),
    PointwiseDef("asinh", 1, "_asinh", promotion=_INT2FLOAT),
    PointwiseDef("acosh", 1, "_acosh", promotion=_INT2FLOAT),
    PointwiseDef("atanh", 1, "_atanh", promotion=_INT2FLOAT),
    # logit's eps is `float?`: omitted means NO clamping, which aten spells as a negative
    # eps sentinel inside its kernel. optional_defaults reproduces that exactly, so one
    # row serves both overloads. (Before optional scalars were modelled this row ignored an
    # explicit eps and silently returned the unclamped result -- nan where aten clamps.)
    PointwiseDef(
        "logit",
        1,
        "_logit",
        promotion=_INT2FLOAT,
        scalars=("eps",),
        optional_defaults={"eps": lambda dt: -1.0},
    ),
    # --- activations (INT_TO_FLOAT like aten; Scalar-parameterized ones bake them) ---
    PointwiseDef("silu", 1, "_silu", promotion=_INT2FLOAT),
    PointwiseDef(
        "elu",
        1,
        "_elu",
        scalars=("alpha", "scale", "input_scale"),
        promotion=_INT2FLOAT,
    ),
    PointwiseDef("celu", 1, "_celu", scalars=("alpha",), promotion=_INT2FLOAT),
    PointwiseDef("selu", 1, "_selu", promotion=_INT2FLOAT),
    PointwiseDef("mish", 1, "_mish", promotion=_INT2FLOAT),
    PointwiseDef(
        "softplus",
        1,
        "_softplus",
        scalars=("beta", "threshold"),
        scalar_defaults=(1, 20),
        promotion=_INT2FLOAT,
    ),
    PointwiseDef(
        "hardtanh",
        1,
        "_hardtanh",
        scalars=("min_val", "max_val"),
        scalar_defaults=(-1, 1),
    ),
    PointwiseDef("hardsigmoid", 1, "_hardsigmoid", promotion=_INT2FLOAT),
    PointwiseDef("relu6", 1, "_relu6"),
    PointwiseDef("threshold", 1, "_threshold", scalars=("threshold", "value")),
    PointwiseDef(
        "hardshrink", 1, "_hardshrink", scalars=("lambd",), scalar_defaults=(0.5,)
    ),
    PointwiseDef(
        "softshrink", 1, "_softshrink", scalars=("lambd",), scalar_defaults=(0.5,)
    ),
    # --- comparisons (ALWAYS_BOOL: output is bool) ---
    PointwiseDef("gt.Tensor", 2, "_gt", promotion=_BOOL),
    PointwiseDef("lt.Tensor", 2, "_lt", promotion=_BOOL),
    PointwiseDef("ge.Tensor", 2, "_ge", promotion=_BOOL),
    PointwiseDef("le.Tensor", 2, "_le", promotion=_BOOL),
    PointwiseDef("eq.Tensor", 2, "_eq", promotion=_BOOL),
    PointwiseDef("ne.Tensor", 2, "_ne", promotion=_BOOL),
    # logical_* / signbit / isnan-family also produce Boolean outputs.
    PointwiseDef("logical_and", 2, "_logical_and", promotion=_BOOL),
    PointwiseDef("logical_or", 2, "_logical_or", promotion=_BOOL),
    PointwiseDef("logical_xor", 2, "_logical_xor", promotion=_BOOL),
    PointwiseDef("logical_not", 1, "_logical_not", promotion=_BOOL),
    PointwiseDef("signbit", 1, "_signbit", promotion=_BOOL),
    PointwiseDef("isnan", 1, "_isnan", promotion=_BOOL),
    PointwiseDef("isinf", 1, "_isinf", promotion=_BOOL),
    PointwiseDef("isfinite", 1, "_isfinite", promotion=_BOOL),
    # --- ternary ---
    PointwiseDef("addcmul", 3, "_addcmul", scalars=("value",)),
    # --- trivial additions (math in ops.py; semantics verified against aten) ---
    # sgn on REAL input is bit-identical to sign (aten routes real sgn to sign_stub and
    # only special-cases complex), so it reuses the existing fn -- no new math.
    PointwiseDef("sgn", 1, "_sign"),
    PointwiseDef("angle", 1, "_angle", promotion=_INT2FLOAT),
    PointwiseDef("isposinf", 1, "_isposinf", promotion=_BOOL),
    PointwiseDef("isneginf", 1, "_isneginf", promotion=_BOOL),
    PointwiseDef("sinc", 1, "_sinc", promotion=_INT2FLOAT),
    PointwiseDef(
        "heaviside",
        2,
        "_heaviside",
        precondition=_all_same_dtype,
    ),
    PointwiseDef("logaddexp2", 2, "_logaddexp2"),
    PointwiseDef("special_entr", 1, "_entr", promotion=_INT2FLOAT),
    PointwiseDef("special_xlog1py", 2, "_xlog1py", promotion=_INT2FLOAT),
    PointwiseDef("hardswish", 1, "_hardswish"),
    # aten's default negative_slope is 0.01: without scalar_defaults a defaulted call
    # would bake 1 and compute plain identity.
    PointwiseDef(
        "leaky_relu",
        1,
        "_leaky_relu",
        scalars=("negative_slope",),
        scalar_defaults=(0.01,),
    ),
    PointwiseDef("addcdiv", 3, "_addcdiv", scalars=("value",)),
    # --- easy tier: mode-selected fn / optional scalars (new PointwiseDef fields) ---
    # gelu is ONE aten overload wrapping TWO kernels, chosen by a string kwarg.
    PointwiseDef(
        "gelu",
        1,
        "_gelu_erf",  # the mode map supplies the real fn; this is the "none" default
        mode_kwarg="approximate",
        mode_fns={None: "_gelu_erf", "none": "_gelu_erf", "tanh": "_gelu_tanh"},
    ),
    # Floating floor/trunc divide remains on ATen; only true division is served.
    PointwiseDef(
        "div.Tensor_mode",
        2,
        "_div",
        mode_kwarg="rounding_mode",
        mode_fns={None: "_div"},
        promotion=_INT2FLOAT,
    ),
    # clamp: BOTH bounds optional and independently omittable. An omitted bound fills
    # with the matching infinity, so the one NaN-propagating formula covers every arity.
    PointwiseDef(
        "clamp",
        1,
        "_clamp",
        scalars=("min", "max"),
        optional_defaults={"min": _lowest, "max": _highest},
    ),
    # nan_to_num: three optional floats whose defaults are DTYPE-dependent (an omitted
    # posinf is that dtype's finite max -- fp16 65504, fp32 3.4e38), which fixed
    # scalar_defaults cannot express.
    PointwiseDef(
        "nan_to_num",
        1,
        "_nan_to_num",
        scalars=("nan", "posinf", "neginf"),
        optional_defaults={
            "nan": lambda dt: 0.0,
            "posinf": lambda dt: torch.finfo(dt).max,
            "neginf": lambda dt: -torch.finfo(dt).max,
        },
    ),
    PointwiseDef("lerp.Scalar", 2, "_lerp", scalars=("weight",)),
    PointwiseDef("lerp.Tensor", 3, "_lerp", precondition=_lerp_dtypes_ok),
)


def variant_aten_name(base_aten: str, variant: PointwiseVariant) -> str | None:
    """aten overload name for (base functional row, variant), or None if aten lacks it.

    Derived by RULE from the functional base (no per-op table), matching aten's naming
    verified against the dispatcher:
      functional : the base name unchanged ("add.Tensor", "neg").
      .out       : base op + "_out" on an explicit overload ("add.Tensor" -> "add.out"?
                   no -- overload-qualified ops append "_out": "gt.Tensor" -> "gt.Tensor_out";
                   a bare/`.Tensor` arithmetic op uses ".out": "add.Tensor" -> "add.out",
                   "neg" -> "neg.out"). We probe both and keep the one aten defines.
    Returns a fully overload-qualified "op.overload" string (what register_op_override
    and _resolve_aten_overload expect), or None to skip this variant for this op.
    """
    op, _, ov = base_aten.partition(".")
    ov = ov or "default"
    if variant.overload_suffix == "":
        return base_aten
    if variant.overload_suffix == "out":
        # aten spells .out either as "<op>.out" (arithmetic: add.Tensor -> add.out) or
        # "<op>.<ov>_out" (comparisons: gt.Tensor -> gt.Tensor_out). Probe both, keep
        # whichever aten defines -- no per-op table needed.
        return _first_defined([f"{op}.out", f"{op}.{ov}_out"])
    return None


def _first_defined(candidates: list[str]) -> str | None:
    # Return the first candidate aten actually defines, in REGISTRATION form, else None.
    # Registration/get_kernel take a bare op name for the default overload ("neg_", not
    # "neg_.default") and the qualified "op.overload" otherwise ("add_.Tensor") -- same
    # convention the functional rows use. Kept here (not overrides.py) so the name
    # derivation lives beside the variant DATA; torch is imported, no cutlass touched.
    for name in candidates:
        o, _, ov = name.partition(".")
        try:
            getattr(getattr(torch.ops.aten, o), ov or "default")
        except AttributeError:
            continue
        return o if ov in ("", "default") else name
    return None
