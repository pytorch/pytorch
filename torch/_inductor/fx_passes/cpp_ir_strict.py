"""Numerical/capability policy for the optional C++ LoopsIR path.

This module is the single place that decides which stock FX rewrites are
bit-preserving and which must be skipped when
``torch._inductor.config.experimental_cpp_ir`` is enabled, and exposes the
validated numerical-policy reader reused by the frontend (decomposition
filter, FX pass gates) and the backend (Program/TileOptions threading).

Modes (``torch._inductor.config.experimental_cpp_ir_numerics``; validated
only when the C++ lane is enabled, never on the stock path):

- ``strict_eager`` (default): preserve eager bits. Every FP-changing
  transformation is skipped and numerically sensitive ops (GELU, softmax,
  mean/var/std, layer/group norm, ...) are kept as original native extern
  boundaries (their decompositions would change rounding).
- ``inductor``: follow stock Inductor's effective numerical choices (FP
  fusion controlled by ``emulate_precision_casts``, FTZ by
  ``eager_numerics.disable_ftz``). Stock numerical rewrites are restored
  wherever the lane has an executable lowering (generated pointwise Triton)
  or a validated native extern (exact eager execution).

Capability exclusions apply in BOTH modes: native pooling decompositions
(``max_pool2d/3d_with_indices`` require stock lowerings), matmul-to-pointwise
decompositions (different accumulation order; the lane keeps native GEMM
externs), the backend-private online-softmax rewrite (its
``prepare_softmax_online`` prim is documented Triton-kernel-only -- never
enable it in the lane without a validated executable native fallback), and
QUACK/template-only rewrites.

Scope contract (worker-owned, orchestrator owns backend/lowering):
- Default ``experimental_cpp_ir=False`` MUST NOT change stock behavior. Every
  gate below returns stock values when the flag is off, and the numerics
  option is neither read nor validated off-lane.
- When the flag is on, the normal AOT + post-grad pipeline still runs
  (``use_post_grad_passes`` / ``use_pre_grad_passes`` are NOT disabled
  wholesale). Only individual FP-changing transformations are skipped so the
  graph keeps original eager bits.
- ``x * 1`` / ``x + 0`` (and ``x - 0`` / ``x / 1``) MUST be retained: they
  change signed-zero / NaN payload rounding, so neutral-op elimination is
  unsafe under this option.
- Unproven numeric library ops (e.g. BatchNorm / cuDNN boundaries) are kept
  opaque where the explicit decomp table allows it. The stock AOT
  ``py_impl(Autograd)`` -> ``native_batch_norm`` chain fires independently of
  the explicit table, so ``{}`` alone does NOT preserve the cuDNN boundary;
  disabling functionalization alone also does NOT. There is no existing scoped
  ``compile_fx`` API to preserve that boundary without a broader framework
  change (pre_dispatch IR is export-only). Callers must treat BN as blocked
  and the helper/backend must reject rather than claim bit-correctness.
- The ManualBERT-style normalization (sample std with correction N-1, eps
  added outside the sqrt) is NOT equivalent to biased-variance LayerNorm;
  no pass in this lane may rewrite that pattern into a regular layernorm.

New module name coordinated with orchestrator: this file
(``torch._inductor.fx_passes.cpp_ir_strict``) is the strict-frontend helper.
The backend helper contract (implemented by orchestrator) is::

    from .loops_ir_backend import compile_fx_loops_ir_output

    return compile_fx_loops_ir_output(
        gm,
        example_inputs,
        graph_kwargs=graph_kwargs,
        inputs_to_check=inputs_to_check,
        runnable_graph_str=runnable_graph_str,
        inductor_post_grad_graph_str=inductor_post_grad_graph_str,
    )

The backend helper owns C++ optimization/emission, ``CompiledFxGraph``
construction, and metrics, receiving full ``graph_kwargs`` flags.
"""

from __future__ import annotations

from typing import Any


# Valid ``experimental_cpp_ir_numerics`` modes.
STRICT_EAGER_NUMERICS = "strict_eager"
INDUCTOR_NUMERICS = "inductor"
_VALID_NUMERICS: frozenset[str] = frozenset({STRICT_EAGER_NUMERICS, INDUCTOR_NUMERICS})


def cpp_ir_enabled() -> bool:
    """True when the optional C++ lane is enabled, in either numerical mode.

    Capability gates (native pooling, backend-private/template-only rewrites)
    use this: the lane cannot execute those in ``inductor`` mode either.
    """
    from torch._inductor import config

    return bool(config.experimental_cpp_ir)


def cpp_ir_numerics() -> str:
    """Validated numerics mode for the C++ lane (frontend/backend shared).

    Returns ``"strict_eager"`` when the lane is off (the option is meaningless
    off-lane and is deliberately NOT validated there, so stock behavior never
    depends on it). When the lane is on, unknown modes raise ``ValueError``.
    """
    from torch._inductor import config

    if not cpp_ir_enabled():
        return STRICT_EAGER_NUMERICS
    mode = config.experimental_cpp_ir_numerics
    if mode not in _VALID_NUMERICS:
        raise ValueError(
            "unknown experimental_cpp_ir_numerics "
            f"'{mode}' (expected one of {sorted(_VALID_NUMERICS)}); "
            "the C++ lane supports only 'strict_eager' and 'inductor'"
        )
    return mode


def cpp_ir_strict_enabled() -> bool:
    """True only when the lane is enabled AND strict-eager numerics hold.

    Numerical gates (neutral-op elimination, reciprocal-sqrt, addmm fusion,
    GELU/softmax/mean/std decompositions, ...) use this: ``inductor`` mode
    restores the stock rewrite wherever the lane can execute it.
    """
    return cpp_ir_enabled() and cpp_ir_numerics() == STRICT_EAGER_NUMERICS


def keep_fp_neutral_op_elimination() -> bool:
    """Whether ``x*1`` / ``x+0`` style rewrites are allowed.

    Stock policy eliminates them; strict policy retains them to preserve
    NaN / signed-zero rounding. Default (flag off) keeps stock behavior.
    """
    return not cpp_ir_strict_enabled()


def allow_online_softmax_rewrite() -> bool:
    """Whether the online-softmax graph rewrite may run.

    Capability gate: the replacement funnels through the backend-private
    ``prepare_softmax_online`` prim (documented Triton-kernel-only). The C++
    lane has no validated executable lowering or native fallback for it, so
    it stays blocked in BOTH modes until such a fallback is proven.
    """
    return not cpp_ir_enabled()


def allow_quack_symmetric_mm() -> bool:
    """Whether the QUACK symmetric-MM replacement may run.

    Capability gate: template-only rewrite with no validated lowering in the
    C++ lane; blocked in BOTH modes initially.
    """
    return not cpp_ir_enabled()


def allow_reciprocal_sqrt_to_rsqrt() -> bool:
    """Whether ``reciprocal(sqrt(x)) -> rsqrt(x)`` may run.

    Two roundings vs one rounding differ, so strict keeps the original.
    """
    return not cpp_ir_strict_enabled()


def allow_addmm_fusion_or_unfuse() -> bool:
    """Whether addmm fuse / unfuse rewrites may run."""
    return not cpp_ir_strict_enabled()


def allow_addcdiv_fma_fusion() -> bool:
    """Whether ``inp + (t1/t2)*value -> addcdiv`` FMA fusion may run."""
    return not cpp_ir_strict_enabled()


def allow_scaled_mm_output_scale_fold() -> bool:
    """Whether scalar-mul-into-``_scaled_mm.scale_result`` may run."""
    return not cpp_ir_strict_enabled()


def allow_pointless_cumsum_rewrite() -> bool:
    """Whether ``cumsum(full)`` pointwise rewrite may run."""
    return not cpp_ir_strict_enabled()


def allow_scaled_softmax_stabilization() -> bool:
    """Whether ``scale(x) - amax(scale(x))`` stabilization may run."""
    return not cpp_ir_strict_enabled()


def allow_bmm_to_mm() -> bool:
    """Whether ``bmm(batch=1) -> mm`` may run.

    Different kernel selection can change accumulation order, so strict
    retains the original op.
    """
    return not cpp_ir_strict_enabled()


def allow_zero_bias_removal() -> bool:
    """Whether all-zero efficient-attention bias removal may run."""
    return not cpp_ir_strict_enabled()


def allow_var_std_dedup() -> bool:
    """Whether var/std reduction dedup may run (already opt-in off)."""
    if cpp_ir_strict_enabled():
        return False
    from torch._inductor import config

    return bool(config.var_std_reduction_dedup)


# Capability exclusions: the C++ lane cannot execute these decompositions in
# EITHER numerical mode, so they are filtered whenever the lane is enabled.
#
# - Native pooling (max_pool2d/3d_with_indices): their decompositions require
#   stock Inductor lowerings; the lane keeps the native pooling op as an
#   exact eager extern instead.
# - Matmul/bmm/addmm (all overloads): decomposing GEMM into pointwise
#   mul+sum changes accumulation order versus both eager and stock Inductor;
#   the lane keeps the native GEMM extern in both modes.
CAPABILITY_EXCLUDE_DECOMP_NAMES: frozenset[str] = frozenset(
    {
        "aten.max_pool2d_with_indices.default",
        "aten.max_pool2d_with_indices.out",
        "aten.max_pool3d_with_indices.default",
        "aten.max_pool3d_with_indices.out",
        "aten.mm.default",
        "aten.mm.out",
        "aten.mm.dtype",
        "aten.mm.dtype_out",
        "aten.bmm.default",
        "aten.bmm.out",
        "aten.bmm.dtype",
        "aten.bmm.dtype_out",
        "aten.addmm.default",
        "aten.addmm.out",
        "aten.addmm.dtype",
        "aten.addmm.dtype_out",
    }
)


# Strict-only numerical exclusions: decompositions that change FP rounding
# versus eager (or versus the non-FMA generated kernels) and must stay opaque
# as native extern boundaries under strict numerics. In ``inductor`` mode
# these follow stock: the decomposed pointwise ops execute through the lane's
# generated Triton lowering with stock-effective flags.
#
# Kept as string names so decomposition.py can filter without importing
# op objects at module load (avoids circulars in unit tests).
STRICT_NUMERICAL_EXCLUDE_DECOMP_NAMES: frozenset[str] = frozenset(
    {
        # Pointwise rounding changes (extra rounding / FMA vs separate).
        "aten.silu.default",
        "aten.silu.out",
        "aten.lerp.Scalar",
        "aten.lerp.Tensor",
        "aten._foreach_lerp.Scalar",
        "aten._foreach_lerp.ScalarList",
        "aten._softmax_backward_data.default",
        "aten.round.decimals",
        # GELU: tanh/erf approximations round differently than eager's exact
        # kernel; strict keeps the native op as an extern boundary.
        "aten.gelu.default",
        "aten.gelu.out",
        # Softmax family: online/stable reorderings change rounding; strict
        # keeps the native ops (externs execute exact eager softmax).
        "aten.softmax.int",
        "aten.log_softmax.int",
        "aten._softmax.default",
        "aten._softmax.out",
        "aten._log_softmax.default",
        "aten._log_softmax.out",
        "aten._safe_softmax.default",
        "aten.special_softmax.default",
        "aten.special_log_softmax.default",
        # Variance/std: eager-biased-vs-corrected formulas round differently
        # (notably the ManualBERT N-1 sample-std + outside-sqrt eps pattern);
        # strict keeps native mean/var/std extern boundaries. (aten.mean has
        # no decomposition entry, so it stays native by construction.)
        "aten.var.default",
        "aten.var.dim",
        "aten.std.default",
        "aten.std.dim",
        "aten.std.correction",
        "aten.std.correction_out",
        "aten.std.out",
        "aten.std_mean.default",
        "aten.std_mean.dim",
        "aten.std_mean.correction",
        "aten.std_mean.correction_out",
        "aten.var_mean.default",
        "aten.var_mean.dim",
        # Normalization library boundaries: keep opaque so the backend can
        # reject rather than silently generate non-bit-correct pointwise
        # code. Note the stock AOT py_impl chain still decomposes
        # native_batch_norm -> _native_batch_norm_legit independently of this
        # table (blocker).
        "aten.native_batch_norm.default",
        "aten._native_batch_norm_legit.default",
        "aten._native_batch_norm_legit.no_stats",
        "aten._native_batch_norm_legit_no_training.default",
        "aten._native_batch_norm_legit_functional.default",
        "aten._batch_norm_with_update.default",
        "aten._batch_norm_with_update_functional.default",
        "aten._batch_norm_no_update.default",
        "aten.batch_norm_backward.default",
        "aten.native_batch_norm_backward.default",
        "aten.native_batch_norm_backward.out",
        "aten.cudnn_batch_norm.default",
        "aten.cudnn_batch_norm_backward.default",
        "aten.miopen_batch_norm.default",
        "aten.miopen_batch_norm_backward.default",
        "aten.native_group_norm.default",
        "aten.layer_norm.default",
        "aten.native_layer_norm.default",
        "aten.native_layer_norm.out",
        "aten.native_layer_norm_backward.default",
        "aten.native_layer_norm_backward.out",
        "aten.group_norm.default",
    }
)


# Backwards-compatible union (strict mode excludes both sets). Prefer the
# split names above for new code.
STRICT_EXCLUDE_DECOMP_NAMES: frozenset[str] = (
    CAPABILITY_EXCLUDE_DECOMP_NAMES | STRICT_NUMERICAL_EXCLUDE_DECOMP_NAMES
)


def filter_strict_decompositions(table: dict[Any, Any]) -> dict[Any, Any]:
    """Return a copy of ``table`` without lane-excluded entries.

    Stock path (lane off) returns ``table`` unchanged and never validates the
    numerics option. Lane on + ``inductor`` removes only the capability set;
    lane on + ``strict_eager`` additionally removes the strict numerical set.
    Unknown policies raise. Matching is by ``str(op)`` so both
    OpOverloadPacket and OpOverload keys are covered.
    """
    if not cpp_ir_enabled():
        return table
    mode = cpp_ir_numerics()  # validates; raises on unknown policies
    if mode == INDUCTOR_NUMERICS:
        excluded = CAPABILITY_EXCLUDE_DECOMP_NAMES
    else:
        excluded = STRICT_EXCLUDE_DECOMP_NAMES
    filtered: dict[Any, Any] = {}
    for op, fn in table.items():
        try:
            name = str(op)
        except Exception:
            filtered[op] = fn
            continue
        # OpOverload str is like "aten.silu.default"; packets lack overload.
        if name in excluded:
            continue
        filtered[op] = fn
    return filtered
