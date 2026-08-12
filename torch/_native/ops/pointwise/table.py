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
#   fn(*input_vals, *scalar_consts) -> result | tuple-of-results
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
    nout: int = 1  # number of outputs (>1: e.g. frexp)
    # ESCAPE HATCH for ops whose output dtypes are NOT all the promotion result
    # (e.g. frexp -> (float mantissa, int32 exponent)). Maps the promotion result
    # torch dtype -> list[torch dtype] of length nout. None -> every output uses the
    # promotion result dtype (the common case).
    out_dtypes: Callable | None = None
    # Restrict the INPUT dtypes this override serves; inputs outside the set fall
    # back to aten. None -> the family default (all supported floats). Use to narrow
    # an op whose kernel is only correct for some dtypes (e.g. frexp excludes fp64).
    dtypes: tuple | None = None
    # ADDITIONAL integer input dtypes this op serves, beyond the float default. Empty
    # by default (float-only). Set INT_DTYPES for ops whose @cute.jit fn is correct on
    # INTEGER compute AND whose aten promotion keeps integers integer (verified against
    # eager): the DEFAULT-arithmetic ops (add/sub/mul/neg/max/min/sign/relu/addcmul) and
    # comparisons (int compute -> bool out). NOT set for:
    #   - transcendentals / div / atan2: aten promotes int -> FLOAT (INT_TO_FLOAT), so
    #     an int input already flows through the float compute path -- see int_via_float;
    #   - floor/ceil/trunc: no-ops on integers in aten, and cute.math.floor rejects Int,
    #     so we decline (aten no-ops it);
    #   - frexp: float-only.
    int_dtypes: tuple = ()
    # True for INT_TO_FLOAT ops (exp/log/sqrt/div/atan2/...): an integer input is valid
    # and promotes to a FLOAT result, so the existing float compute path serves it -- we
    # just have to ACCEPT the int input dtype at the cond. Distinct from int_dtypes
    # (integer-in, integer-out); an op sets at most one.
    int_via_float: bool = False
    # Alternate ops.py fn used when the COMPUTE dtype is integer, for ops whose integer
    # math differs from their float math (fmod/remainder: trunc/floor via cute.math.floor
    # is float-only, the int versions use the DSL's truncating int division). None -> the
    # one `fn` serves all compute dtypes (the common case; +,-,*,&,| are dtype-generic).
    int_fn: str | None = None
    # Skip the .out variant because an ATEN kernel redispatches a WRAPPED-NUMBER tensor
    # into it. aten's pow_Scalar_out builds wrapped_scalar_tensor(base, exp.device()) --
    # a CUDA tensor flagged is_wrapped_number -- and calls at::pow_out (Pow.cpp:66). Our
    # router is a Python kernel, and converting a boxed wrapped-number tensor to Python
    # asserts it is CPU (pybind_utils.cpp toPyObject), so registering the .out overload
    # turns `2.0 ** cuda_tensor` / pow(2.0, t) / float_power(2.0, t) / integer ldexp into
    # an INTERNAL ASSERT. The assert fires in the boxed->Python conversion, BEFORE any
    # cond runs, so it cannot be declined in Python -- the overload must not be
    # registered at all. Functional and in-place still are.
    skip_out_variant: bool = False
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
    # Select the ops.py fn by a STRING kwarg rather than by dtype (int_fn's axis).
    # aten spells two distinct kernels as one overload plus a mode string: gelu's
    # `approximate` in {"none","tanh"} and div's `rounding_mode` in {None,"floor",
    # "trunc"}. Maps kwarg VALUE -> fn name; the kwarg's own name is `mode_kwarg`.
    # A value absent from the map declines (aten validates it and raises).
    mode_kwarg: str | None = None
    mode_fns: dict | None = None
    # Per-mode promotion override, for ops whose mode changes the type-promotion rule:
    # div(rounding_mode=None) is true division (INT_TO_FLOAT) while the floor/trunc
    # modes keep integers integral (DEFAULT). Maps mode value -> PromotionKind; unset
    # modes use `promotion`.
    mode_promotion: dict | None = None
    # int_fn's per-mode counterpart: the Int-compute fn for a mode whose integer math
    # differs from its float math (div floor/trunc -- the DSL's `//` floors, and
    # cute.math.floor rejects Int). Maps mode value -> fn name; falls back to mode_fns.
    mode_int_fns: dict | None = None


_DEFAULT = PromotionKind.DEFAULT
_INT2FLOAT = PromotionKind.INT_TO_FLOAT
_BOOL = PromotionKind.ALWAYS_BOOL

# Integer input dtypes the int-capable ops additionally serve (int_dtypes / the int-in
# side of comparisons). int32/int64 only: torch2cute has no int8/int16, and bool
# arithmetic is excluded (aten's neg(bool) even raises), so we stay on the two integer
# widths the kernel + launch glue already support.
_INT_DTYPES = (torch.int32, torch.int64)


def _lowest(dt):
    # An omitted clamp lower bound: -inf for float, the type minimum for int.
    return -math.inf if dt.is_floating_point else torch.iinfo(dt).min


def _highest(dt):
    return math.inf if dt.is_floating_point else torch.iinfo(dt).max


class PointwiseVariant(NamedTuple):
    # One mutation VARIANT of a pointwise op (functional / .out / in-place), as DATA so
    # the generic registration (overrides.py) derives all three from one PointwiseDef row
    # without per-op code. aten semantics for each variant were verified against eager
    # (see the derisk sweeps); the fields below encode exactly those rules:
    #
    #   name             identifies the variant for messages / keys.
    #   overload_suffix  how aten names this overload relative to the functional base:
    #                    "" (functional), "out" (.out variant -> base+"_out" or ".out"),
    #                    "inplace" (op_ with the same overload). Resolved by _variant_op.
    #   out_from         where the OUTPUT tensor(s) come from, and thus the out DTYPE:
    #                    "alloc"  -> allocate fresh, dtype = promotion result (functional);
    #                    "out_kw" -> the `out=` kwarg tensor, keeps ITS OWN dtype (aten
    #                                casts the compute result into it);
    #                    "self"   -> operand 0, in-place; keeps self's dtype (aten
    #                                downcasts the compute result, e.g. f32.add_(f64)->f32,
    #                                and comparisons write 0/1 into self, NOT bool).
    #   shape_rule       broadcast constraint (all variants broadcast operands):
    #                    "free"      -> output = broadcast_shapes(operands) (functional/.out
    #                                   -- .out is resized to it);
    #                    "eq_self"   -> broadcast_shapes(operands) must EQUAL self.shape
    #                                   (in-place cannot grow self; operands broadcast UP
    #                                   to it). Incompatible -> decline, aten raises.
    #   has_inplace      False for ops with no aten in-place (maximum/minimum/frexp): the
    #                    in-place variant is skipped for them.
    name: str
    overload_suffix: str
    out_from: str
    shape_rule: str


# The three variants every eligible pointwise op exposes. Registration walks this tuple
# per row; a row whose op lacks an in-place overload (has_inplace via _variant_op = None)
# simply skips that variant. Adding a new variant class is one entry here, not N ops.
POINTWISE_VARIANTS: tuple[PointwiseVariant, ...] = (
    PointwiseVariant("functional", "", "alloc", "free"),
    PointwiseVariant("out", "out", "out_kw", "free"),
    PointwiseVariant("inplace", "inplace", "self", "eq_self"),
)

POINTWISE_DEF_TABLE: tuple[PointwiseDef, ...] = (
    # --- binary / unary arithmetic (DEFAULT: output dtype = promoted input) ---
    # int_dtypes: integer-in -> integer-out (fn correct on Int compute, verified).
    PointwiseDef("neg", 1, "_neg", int_dtypes=_INT_DTYPES),
    PointwiseDef("add.Tensor", 2, "_add", scalars=("alpha",), int_dtypes=_INT_DTYPES),
    PointwiseDef("sub.Tensor", 2, "_sub", scalars=("alpha",), int_dtypes=_INT_DTYPES),
    PointwiseDef("mul.Tensor", 2, "_mul", int_dtypes=_INT_DTYPES),
    # div.Tensor is INT_TO_FLOAT in aten (true division: int/int -> float, and it is
    # this promotion even for float inputs). int_via_float accepts the int input, which
    # then flows the float compute path to a float result.
    PointwiseDef("div.Tensor", 2, "_div", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("maximum", 2, "_maximum", int_dtypes=_INT_DTYPES),
    PointwiseDef("minimum", 2, "_minimum", int_dtypes=_INT_DTYPES),
    PointwiseDef("atan2", 2, "_atan2", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("rsub.Tensor", 2, "_rsub", scalars=("alpha",), int_dtypes=_INT_DTYPES),
    # fmax/fmin are the NaN-SUPPRESSING pair (vs maximum/minimum which propagate).
    PointwiseDef("fmax", 2, "_fmax", int_dtypes=_INT_DTYPES),
    PointwiseDef("fmin", 2, "_fmin", int_dtypes=_INT_DTYPES),
    PointwiseDef("clamp_min.Tensor", 2, "_clamp_min", int_dtypes=_INT_DTYPES),
    PointwiseDef("clamp_max.Tensor", 2, "_clamp_max", int_dtypes=_INT_DTYPES),
    # atan2/copysign/xlogy/xlog1py are INT_TO_FLOAT in aten just like div: an all-integer
    # operand pair promotes to float32. int_via_float alone only ACCEPTS the int input --
    # without the promotion kind, DEFAULT promotion keeps the compute dtype integer and
    # cute.math.log/atan (float-only primitives) reject it (copysign's math happens to
    # lower on ints, so it silently returned an integer instead of aten's float32).
    PointwiseDef(
        "copysign.Tensor", 2, "_copysign", promotion=_INT2FLOAT, int_via_float=True
    ),
    PointwiseDef("hypot", 2, "_hypot"),
    PointwiseDef("logaddexp", 2, "_logaddexp"),
    PointwiseDef("xlogy.Tensor", 2, "_xlogy", promotion=_INT2FLOAT, int_via_float=True),
    # pow keeps integers integer (aten int^int -> int); the float fn is the exp2/log2
    # composite with full sign/edge handling. Integer pow needs a repeated-multiply
    # loop the DSL can't express over a runtime exponent -> float-only.
    PointwiseDef("pow.Tensor_Tensor", 2, "_pow", skip_out_variant=True),
    # fmod/remainder/floor_divide: integer math differs from float (truncating int
    # division vs cute.math.floor) -> int_fn picks the Int-compute variant.
    PointwiseDef("fmod.Tensor", 2, "_fmod", int_dtypes=_INT_DTYPES, int_fn="_fmod_int"),
    PointwiseDef(
        "remainder.Tensor",
        2,
        "_remainder",
        int_dtypes=_INT_DTYPES,
        int_fn="_remainder_int",
    ),
    PointwiseDef(
        "floor_divide",
        2,
        "_floor_divide",
        int_dtypes=_INT_DTYPES,
        int_fn="_floor_divide_int",
    ),
    # --- rounding / sign / activation (DEFAULT) ---
    # floor/ceil/trunc are no-ops on integers in aten AND cute.math.floor rejects Int, so
    # they stay float-only (an int input declines -> aten no-ops it). sign/relu are correct
    # on Int compute.
    PointwiseDef("floor", 1, "_floor"),
    PointwiseDef("ceil", 1, "_ceil"),
    PointwiseDef("trunc", 1, "_trunc"),
    PointwiseDef("sign", 1, "_sign", int_dtypes=_INT_DTYPES),
    PointwiseDef("relu", 1, "_relu", int_dtypes=_INT_DTYPES),
    PointwiseDef("abs", 1, "_abs", int_dtypes=_INT_DTYPES),
    PointwiseDef("square", 1, "_square", int_dtypes=_INT_DTYPES),
    # round/frac: no-op / undefined on integers like floor (round(int)=int in aten);
    # float-only for the same cute.math.floor reason.
    PointwiseDef("round", 1, "_round"),
    PointwiseDef("frac", 1, "_frac"),
    PointwiseDef("deg2rad", 1, "_deg2rad", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("rad2deg", 1, "_rad2deg", promotion=_INT2FLOAT, int_via_float=True),
    # --- unary transcendental math (INT_TO_FLOAT: int input -> float output) ---
    PointwiseDef("exp", 1, "_exp", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("exp2", 1, "_exp2", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("expm1", 1, "_expm1", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("log", 1, "_log", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("log2", 1, "_log2", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("log10", 1, "_log10", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("log1p", 1, "_log1p", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("sqrt", 1, "_sqrt", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("rsqrt", 1, "_rsqrt", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef(
        "reciprocal", 1, "_reciprocal", promotion=_INT2FLOAT, int_via_float=True
    ),
    PointwiseDef("sin", 1, "_sin", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("cos", 1, "_cos", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("tan", 1, "_tan", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("asin", 1, "_asin", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("acos", 1, "_acos", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("atan", 1, "_atan", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("tanh", 1, "_tanh", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("erf", 1, "_erf", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("erfc", 1, "_erfc", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("sigmoid", 1, "_sigmoid", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("cosh", 1, "_cosh", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("sinh", 1, "_sinh", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("asinh", 1, "_asinh", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("acosh", 1, "_acosh", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("atanh", 1, "_atanh", promotion=_INT2FLOAT, int_via_float=True),
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
    # --- comparisons (ALWAYS_BOOL: output is bool) -- integer-in -> bool-out ---
    PointwiseDef("gt.Tensor", 2, "_gt", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    PointwiseDef("lt.Tensor", 2, "_lt", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    PointwiseDef("ge.Tensor", 2, "_ge", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    PointwiseDef("le.Tensor", 2, "_le", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    PointwiseDef("eq.Tensor", 2, "_eq", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    PointwiseDef("ne.Tensor", 2, "_ne", promotion=_BOOL, int_dtypes=_INT_DTYPES),
    # logical_* / signbit / isnan-family are ALWAYS_BOOL over any input kind.
    PointwiseDef(
        "logical_and", 2, "_logical_and", promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "logical_or", 2, "_logical_or", promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "logical_xor", 2, "_logical_xor", promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "logical_not", 1, "_logical_not", promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef("signbit", 1, "_signbit", promotion=_BOOL),
    PointwiseDef("isnan", 1, "_isnan", promotion=_BOOL),
    PointwiseDef("isinf", 1, "_isinf", promotion=_BOOL),
    PointwiseDef("isfinite", 1, "_isfinite", promotion=_BOOL),
    # --- bitwise (integer-ONLY compute: dtypes overrides the float default away) ---
    PointwiseDef("bitwise_and.Tensor", 2, "_bitwise_and", dtypes=_INT_DTYPES),
    PointwiseDef("bitwise_or.Tensor", 2, "_bitwise_or", dtypes=_INT_DTYPES),
    PointwiseDef("bitwise_xor.Tensor", 2, "_bitwise_xor", dtypes=_INT_DTYPES),
    PointwiseDef("bitwise_not", 1, "_bitwise_not", dtypes=_INT_DTYPES),
    PointwiseDef(
        "bitwise_left_shift.Tensor", 2, "_bitwise_left_shift", dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "bitwise_right_shift.Tensor", 2, "_bitwise_right_shift", dtypes=_INT_DTYPES
    ),
    # --- ternary / multi-output ---
    PointwiseDef("addcmul", 3, "_addcmul", scalars=("value",)),
    PointwiseDef(
        "frexp.Tensor",
        1,
        "_frexp",
        nout=2,
        # (mantissa: promotion-result float, exponent: int32) -- the escape hatch.
        out_dtypes=lambda compute: [compute, torch.int32],
        # log2-derived frexp is exact only for fp16/bf16/fp32; fp64 needs bit
        # extraction (deferred). fp64 falls back to aten.
        dtypes=(torch.float16, torch.bfloat16, torch.float32),
    ),
    # --- trivial additions (math in ops.py; semantics verified against aten) ---
    # sgn on REAL input is bit-identical to sign (aten routes real sgn to sign_stub and
    # only special-cases complex), so it reuses the existing fn -- no new math.
    PointwiseDef("sgn", 1, "_sign", int_dtypes=_INT_DTYPES),
    PointwiseDef("angle", 1, "_angle", promotion=_INT2FLOAT),
    PointwiseDef("isposinf", 1, "_isposinf", promotion=_BOOL),
    PointwiseDef("isneginf", 1, "_isneginf", promotion=_BOOL),
    # aten's sinc also accepts complex (we have no DSL complex), so pin the served
    # dtypes to the float set + ints-via-float rather than inheriting the default.
    PointwiseDef("sinc", 1, "_sinc", promotion=_INT2FLOAT, int_via_float=True),
    PointwiseDef("heaviside", 2, "_heaviside", int_dtypes=_INT_DTYPES),
    PointwiseDef("logaddexp2", 2, "_logaddexp2"),
    PointwiseDef("special_entr", 1, "_entr", promotion=_INT2FLOAT),
    PointwiseDef(
        "special_xlog1py", 2, "_xlog1py", promotion=_INT2FLOAT, int_via_float=True
    ),
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
    # aten REJECTS integer addcdiv ("Integer division with addcdiv is no longer
    # supported"), so this must NOT opt into int inputs -- int_via_float would serve
    # what aten errors on. Float-only, unlike its addcmul twin.
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
    # div's rounding_mode selects the fn AND changes promotion: None is true division
    # (int -> float), floor/trunc keep integers integral. All three fns already exist.
    PointwiseDef(
        "div.Tensor_mode",
        2,
        "_div",
        mode_kwarg="rounding_mode",
        mode_fns={None: "_div", "floor": "_floor_divide", "trunc": "_div_trunc"},
        mode_int_fns={"floor": "_floor_divide_int", "trunc": "_div_trunc_int"},
        mode_promotion={None: _INT2FLOAT, "floor": _DEFAULT, "trunc": _DEFAULT},
        int_dtypes=_INT_DTYPES,
    ),
    # clamp: BOTH bounds optional and independently omittable. An omitted bound fills
    # with the matching infinity, so the one NaN-propagating formula covers every arity.
    PointwiseDef(
        "clamp",
        1,
        "_clamp",
        scalars=("min", "max"),
        optional_defaults={"min": _lowest, "max": _highest},
        int_dtypes=_INT_DTYPES,
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
    PointwiseDef("lerp.Tensor", 3, "_lerp"),
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
      in-place   : op + "_" with the SAME overload ("add.Tensor" -> "add_.Tensor",
                   "neg" -> "neg_.default"). None when aten has no in-place (maximum/
                   minimum/frexp).
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
    if variant.overload_suffix == "inplace":
        return _first_defined([f"{op}_.{ov}", f"{op}_.default"])
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
