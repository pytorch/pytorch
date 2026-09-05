# Copyright (c) 2026, Han Guo, Tri Dao.
"""Ready-to-use fused-GEMM epilogues, written with the @gemm_epilogue fn
contract (quack.epilogue.frontend) on top of the EpiOp library (quack.epilogue.ops).

Each entry is a plain function over the accumulator plus declared resources;
kernels are minted/cached per (fn source, op config, tensor metadata). Pass
tensors via ``mod.gemm(A, B, D, C, epi_args={...}, tile_M=..., ...)``.

Sections:
  * tuple-polymorphic scalar math is shared through ``quack.epi_math``;
  * reusable domain resources live in ``quack.epilogue``: rotary table loads
    in ``rotary`` and per-head RMSNorm statistics in ``head_rmsnorm``.
  * elementwise mods: linear/bias, residual, activation factories, RMS-fused
    (sq-sum reduce), amax (quantization stats), per-tile LSE partials, online
    LSE (stable, plus the CE-eval variant with target-logit gather),
    transformer-block forward (rms_partial -> rstd_swiglu) and the
    rmsnorm-backward link.
  * paired mods (gated / RoPE) and packed-C/D mods (dgated), via unpack/pack.

Numerics of every mod here are pinned by tests/test_gemm_epilogue.py against
torch references.
"""

from __future__ import annotations

import functools

from cutlass import Int32

from torch._vendor.quack import epi_math
from torch._vendor.quack.activation import (
    act_fn_map,
    dact_fn_map,
    dgate_fn_map,
    dgelu_tanh_approx,
    dswiglu,
    dswiglu_oai,
    gate_fn_map,
    gelu_tanh_approx,
    relu,
    relu_sq,
    swiglu,
)
from torch._vendor.quack.epilogue.ops import (
    ColVecLoad,
    ColVecReduce,
    ColVecSelect,
    OnlineLSEReduce,
    RowVecLoad,
    RowVecReduce,
    Scalar,
    TileStore,
)
from torch._vendor.quack.epilogue.head_rmsnorm import HeadRstd
from torch._vendor.quack.epilogue.quantize_out import BlockScaleFactorStore
from torch._vendor.quack.epilogue.rotary import rotary_cos_sin_load
from torch._vendor.quack.epilogue.math import pack, unpack
from torch._vendor.quack.epilogue.frontend import gemm_epilogue


@gemm_epilogue(outputs=("postact",))
def norm_gelu(acc, rstd, weight):
    x = acc * rstd * weight
    return {"D": x, "postact": gelu_tanh_approx(x)}


@gemm_epilogue()
def scaled_residual(acc, c, alpha):
    return {"D": acc * alpha + c}


@gemm_epilogue()
def linear_epi(acc, c, alpha, beta, bias_n, bias_m):
    """The default (linear) epilogue as a mod: alpha*acc + beta*C + rowvec + colvec."""
    return {"D": acc * alpha + c * beta + bias_n + bias_m}


def make_act_mod(fn):
    """Activation-mod factory — exercises closure-salted cache identity."""

    @gemm_epilogue(outputs=("postact",))
    def act_mod(acc):
        return {"D": acc, "postact": fn(acc)}

    return act_mod


relu_mod = make_act_mod(relu)


relu_sq_mod = make_act_mod(relu_sq)


@gemm_epilogue(outputs=("postact",))
def dgelu_mod(acc, c):
    """GemmDAct as a mod: c is the preact, acc is dout; D = dx, postact = act(c)."""
    dx, out = dgelu_tanh_approx(c, acc)
    return {"D": dx, "postact": out}


@gemm_epilogue(outputs=("postact",), reduces={"dbias": RowVecReduce("dbias")})
def dgelu_dbias_mod(acc, c):
    """dgelu_mod + fused bias grad: dbias[n] = sum_m dx, the post-act-grad
    rowvec (the preact bias's gradient)."""
    dx, out = dgelu_tanh_approx(c, acc)
    return {"D": dx, "postact": out, "dbias": dx}


@gemm_epilogue(outputs=("premult",), reduces={"sqsum": ColVecReduce("sqsum", scaled=True)})
def rms_fused(acc, weight):
    """GemmSqReduce as a mod: sqsum accumulated on pre-scale acc, D = acc * weight.
    The scaled reduce returns the factors so the fold is one fma(acc, acc, sum)."""
    return {"D": acc * weight, "sqsum": (acc, acc), "premult": acc}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dswiglu_mod(acc, c):
    """GemmDGated as a mod: acc = dout (per pair), c = packed (x, y) preact.
    Packing is declared by mode and validated against 16-bit C/D at twice GEMM-N."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    return {"D": pack(dx, dy), "postact": out}


def make_dgated_moe_mod(dgate):
    """MoE-expert fc2-dgrad epilogue factory (the gpt-oss shape: biased FFN +
    per-token router score on the expert output). acc = d(expert_out) @ W2^T
    UNSCALED, score colvec = the router weight. Emits:
      * D = packed dpreact with the score folded (dpreact = dgate(x, y, s*dout))
        — both fc1 backward GEMMs stay plain;
      * postact = score-scaled recomputed activation, so fc2-wgrad pairs it
        with the plain dout;
      * dscore[m] = sum_n(postact * unscaled dout) = <expert_out, d(final)>
        per token — the router score grad, one fma via the scaled reduce;
      * the fc1 bias grad as an N-wide rowvec PAIR (the preact is 2N-wide and
        a rowvec slot is one f32): host interleaves db[0::2] = dbias_g,
        db[1::2] = dbias_u, matching the packed (x, y) preact layout."""

    @gemm_epilogue(
        outputs=("postact",),
        ops={"score": ColVecLoad("score")},
        reduces={
            "dscore": ColVecReduce("dscore", scaled=True),
            "dbias_g": RowVecReduce("dbias_g"),
            "dbias_u": RowVecReduce("dbias_u"),
        },
        mode="packed_cd_b16x2",
    )
    def dgated_moe_mod(acc, c, score):
        x, y = unpack(c)
        dx, dy, out = dgate(x, y, acc * score)
        return {
            "D": pack(dx, dy),
            "postact": out * score,
            "dscore": (out, acc),
            "dbias_g": dx,
            "dbias_u": dy,
        }

    return dgated_moe_mod


dswiglu_moe_mod = make_dgated_moe_mod(dswiglu)


# gpt-oss exact: swiglu_oai with the swiglu_limit=7.0 preact clamp (grads
# zeroed where the clamp saturates).
dswiglu_oai_moe_mod = make_dgated_moe_mod(lambda x, y, dout: dswiglu_oai(x, y, dout, limit=7.0))


@gemm_epilogue(outputs=("postact",), reduces={"dsum": ColVecReduce("dsum")}, mode="packed_cd_b16x2")
def dswiglu_dpreact_mod(acc, c):
    """dswiglu_mod + the rstd-correction stat on the raw preact: dsum[m] =
    sum over the full 2N preact dim of dpreact * preact (= dx*x + dy*y per
    pair — dswiglu_rstd_preact_mod's dsum without the deferred rstd). A
    two-product sum, so the scaled (val, scale) fma fold does not apply."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    return {"D": pack(dx, dy), "postact": out, "dsum": dx * x + dy * y}


@gemm_epilogue(
    outputs=("postact",),
    reduces={"dsum": ColVecReduce("dsum", scaled=True)},
    mode="packed_cd_b16x2",
)
def dswiglu_norm_mod(acc, c, rstd):
    """Full dgated: colvec-scaled dout, reduce on unscaled dout (folded as
    fma(out, dout, acc) via the scaled reduce), scaled postact."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc * rstd)
    return {"D": pack(dx, dy), "postact": out * rstd, "dsum": (out, acc)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_mod(acc):
    """GemmGated as a mod: the accumulator pairs over adjacent N because the
    postact buffer is half of GEMM-N."""
    gate, up = unpack(acc)
    return {"postact": swiglu(gate, up)}


@functools.lru_cache(maxsize=None)
def gated_quant_mod(activation):
    """gemm + gated activation + QUANTIZED postact (the MoE FC1 fusion):
    ``postact`` is fp8 e4m3 or packed fp4 values (its dtype picks
    mxfp8/mxfp4/nvfp4 together with the SF dtype), ``postact_sf`` the blocked
    (l?, rm, rk, 32, 4, 4) scale-factor tensor the kernel writes (rk over the
    HALF-width postact N), and the optional ``sfd_norm_const`` scalar folds
    1/per_tensor_scale into the SFs (nvfp4). SM100 only. SF vectors live in
    accumulator space — one vector of ``vec`` postact values spans 2*vec acc
    columns — see BlockScaleFactorStore(output=...)."""
    act = gate_fn_map[activation]

    @gemm_epilogue(
        outputs=(
            TileStore(
                "postact",
                gated=True,
                quant=BlockScaleFactorStore("postact_sf", output="postact"),
            ),
        ),
        extra_ops=(Scalar("sfd_norm_const"),),
        mode="acc_pair",
    )
    def gated_quant_epi(acc):
        gate, up = unpack(acc)
        return {"postact": act(gate, up)}

    return gated_quant_epi


swiglu_quant_mod = gated_quant_mod("swiglu")


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def norm_swiglu_mod(acc, rstd, bias):
    """Gated with rowvec bias (arrives paired) + colvec (scalar), D writeback.
    Pair arithmetic is lane-wise, so the affine part runs before unpacking."""
    v = acc * rstd + bias
    g, u = unpack(v)
    return {"postact": swiglu(g, u), "D": pack(g, u)}


@gemm_epilogue()
def identity_epi(acc):
    """Plain GEMM through the mod path. D's dtype is free, so this is the
    fp32-output wgrad building block: dW(n,k) = dout^T @ x runs as
    ``identity_epi.gemm(dout.mT, x.mT, dW_f32)`` — A arrives M'-major and B
    N'-major (both views, no copies) and the epilogue writes the f32
    accumulator directly (bf16 matmuls can't emit fp32 through torch)."""
    return {"D": acc}


@gemm_epilogue()
def residual_epi(acc, res):
    """Coda Residual: full-tile aux input added to the accumulator."""
    return {"D": acc + res}


@gemm_epilogue(mode="acc_pair")
def rope_epi(acc, table):
    """Coda RoPE: rotate adjacent-N pairs by an interleaved cos/sin table
    ((..., 2j) = cos, (..., 2j+1) = sin), congruent with the D tile."""
    x1, x2 = unpack(acc)
    cos, sin = unpack(table)
    return {"D": pack(x1 * cos - x2 * sin, x1 * sin + x2 * cos)}


@gemm_epilogue(reduces={"sexp": ColVecReduce("sexp")})
def lse_partial_epi(acc, scale):
    """Coda LSE, per-tile flavor: sexp[m, tile] = sum_n exp(acc * scale);
    the host finalizes log(sum(partials)). NOTE: no online max — needs a
    max-combine reduce for large-logit stability (Coda's LSEReduce is online)."""
    return {"D": acc, "sexp": epi_math.exp(acc * scale, fast=True)}


@gemm_epilogue(outs={"lse": OnlineLSEReduce("lse")})
def lse_epi(acc):
    """Logits + stable online (max, sum) LSE partials (l, m, n_tiles). Host:
    lse = logsumexp(partials, -1)."""
    return {"D": acc, "lse": acc}


_lse_target_idx = ColVecLoad("target")


@gemm_epilogue(
    outs={
        "lse": OnlineLSEReduce("lse"),
        "target_logit": ColVecSelect("target_logit", idx_op=_lse_target_idx),
    },
    extra_ops=(_lse_target_idx,),
)
def lse_target_epi(acc):
    """Cross-entropy-eval LM-head epilogue: write the logits, accumulate stable online
    (max, sum) LSE partials (l, m, n_tiles), and gather each row's target-column logit into an
    (l, m) f32 colvec — exact, never rounded through the D dtype (the anchor for the
    fused linear-cross-entropy backward). ``target`` (an (l, m) int32/int64
    colvec in epi_args) feeds the select through its companion load, not the
    fn; rows with out-of-range targets (e.g. ignore_index -100) leave
    target_logit untouched. Host CE forward:
    loss = logsumexp(lse, -1) - target_logit."""
    return {"D": acc, "lse": acc, "target_logit": acc}


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")}, outs={"lse": OnlineLSEReduce("lse")})
def rstd_lse_epi(acc, rstd):
    """LM-head epilogue of a Coda-style transformer stack: apply the final
    boundary's deferred rstd (colvec), write the logits, and accumulate the
    stable online (max, sum) LSE partials (l, m, n_tiles). Host CE forward:
    lse = logsumexp(partials, -1); loss = lse - logits.gather(-1, target)."""
    v = acc * rstd
    return {"D": v, "lse": v}


@gemm_epilogue(reduces={"amax": ColVecReduce("amax", combine="max")})
def amax_epi(acc):
    """Per-tile column amax — the quantized-output (SFD) building block.
    |x| >= 0, so the zero OOB accumulator lanes of a ragged last tile can't
    corrupt the max (see VecReduce.combine note)."""
    return {"D": acc, "amax": epi_math.abs(acc, fast=True)}


def _sq_prepass(acc):
    """Prepass fn: the statistic input, explicit (this replaces epirope's
    _prenorm_vec_ops replay registry — any pre-norm transform would be
    duplicated here in plain sight)."""
    return {"qk": acc * acc}


_head_rstd_op = HeadRstd("qk", eps=1e-6)


@gemm_epilogue(
    ops={"qk": _head_rstd_op},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    extra_ops=(_head_rstd_op.out("rstd_out"),),
)
def head_rmsnorm_epi(acc, qk):
    """Per-head RMSNorm, no weight: qk is the per-(row, head) rstd from the
    HeadRstd statistic (its (head_dim,) host arg only fixes the head width).
    Optional rstd_out (l?, m, n/head_dim): the finalized rstd per (row,
    head), written from the prepass stats — the backward needs it."""
    return {"D": acc * qk}


_qknorm_rstd_op = HeadRstd("qk", eps=1e-6)


@gemm_epilogue(
    ops={"qk": _qknorm_rstd_op, "w": RowVecLoad("w")},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    extra_ops=(_qknorm_rstd_op.out("rstd_out"),),
)
def qknorm_epi(acc, qk, w):
    """Weighted per-head RMSNorm: the rstd statistic and the norm weight are
    independent resources, multiplied in plain sight. w is an ordinary (N,)
    rowvec — pass the head weight repeated per head; qk's host arg fixes
    head_dim (an int, or any tensor whose length is head_dim).
    Optional rstd_out (l?, m, n/head_dim): the finalized rstd per (row,
    head) — the backward needs it."""
    return {"D": acc * qk * w}


@gemm_epilogue(
    ops={
        "cs": rotary_cos_sin_load("cs"),
        "qk": HeadRstd("qk", eps=1e-6),
        "w": RowVecLoad("w"),
    },
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    prepass_mode="element",
    mode="acc_pair",
)
def qk_rope_epi(acc, cs, qk, w):
    """The full epirope composition: per-head RMSNorm (prepass stats x weight
    rowvec) then rotary, in five lines of fn math. TMA table (see
    rotary_cos_sin_load): at the winning clustered-pingpong configs the LDG
    table's register cost is what tips this composition into spills."""
    x1, x2 = unpack(acc * qk * w)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={
        "cs": rotary_cos_sin_load("cs", tma=False),
        "qk": HeadRstd("qk", eps=1e-6),
        "w": RowVecLoad("w"),
    },
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    prepass_mode="element",
    mode="acc_pair",
)
def qk_rope_ldg_epi(acc, cs, qk, w):
    """qk_rope_epi on the gmem->rmem table op (see rope_table_ldg_epi)."""
    x1, x2 = unpack(acc * qk * w)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    outputs=("resid_out",),
    ops={"weight": RowVecLoad("weight")},
    reduces={"sqsum": ColVecReduce("sqsum", scaled=True)},
)
def rms_partial_epi(acc, c, weight):
    """GEMM1 of a block: y = acc + residual(C); write the residual stream (aux),
    the weight-applied output (D — rstd deferred: row scaling commutes through
    the NEXT gemm), and the per-tile sq-sum partials for rstd finalization."""
    y = acc + c
    return {"D": y * weight, "resid_out": y, "sqsum": (y, y)}


@gemm_epilogue(reduces={"dots": ColVecReduce("dots", scaled=True)})
def rms_bwd_partial_epi(acc, y, rstd, w):
    """RMSNorm backward around a dgrad GEMM: acc = dz @ W2^T (= d(norm out)).
    t = acc*w, xhat = saved_prenorm(TileLoad) * rstd; write D = rstd*t and the
    per-tile partials of the correction dot mean(t * xhat)."""
    t = acc * w
    xhat = y * rstd
    return {"D": t * rstd, "dots": (t, xhat)}


# --- Deferred-rstd norm backward (transformer-block bwd, Coda-style) ----------
# Mid-stack boundary: the fwd applied gamma at the producing GEMM (a = h*gamma,
# rstd deferred into the NEXT gemm's epilogue), so the dgrad of a boundary GEMM
# receives acc = da carrying no rstd. dh = da*gamma + residual grad + the
# rstd-gradient correction colvec (host: corr = -(rstd^3/d) * dsum_finalized,
# with dsum from the fc2-dgrad epilogue; qkv side: -(rstd^2/d) * sum(dQKV*QKV)).
# dgamma = sum_m da*h lands as RowVecReduce partials (l, m_tiles, n), finalized
# by a host .sum over tiles.


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "corr": ColVecLoad("corr")},
    reduces={"dw": RowVecReduce("dw", scaled=True)},
)
def rms_bwd_apply_epi(acc, c, y, w, corr):
    """Norm-bwd apply around a dgrad GEMM (mid-stack boundary): acc = da,
    c = incoming residual-stream grad, y = saved residual h (TileLoad),
    w = gamma rowvec, corr colvec. D = dh_total; dw partials = dgamma."""
    return {"D": acc * w + c + y * corr, "dw": (acc, y)}


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "corr": ColVecLoad("corr")},
    reduces={"dw": RowVecReduce("dw", scaled=True)},
)
def rms_bwd_apply_last_epi(acc, y, w, corr):
    """rms_bwd_apply_epi without the residual-grad C: the FINAL boundary
    (lm_head dgrad) — the last residual h_L has no consumer besides the norm."""
    return {"D": acc * w + y * corr, "dw": (acc, y)}


@gemm_epilogue(
    ops={"w": RowVecLoad("w"), "rstd": ColVecLoad("rstd")},
    reduces={
        "dots": ColVecReduce("dots", scaled=True),
        "dw": RowVecReduce("dw", scaled=True),
    },
)
def rms_bwd_entry_epi(acc, c, y, rstd, w):
    """Conventional (non-deferred) rmsnorm bwd around a dgrad GEMM, for the
    ENTRY boundary where the fwd used a standalone full rmsnorm: acc = da with
    a = (y*rstd)*w. Emits D = rstd*t + c (t = acc*w) plus BOTH stats — the
    correction dot partials (finalized and applied by a terminal elementwise
    pass: dh = D - xhat*rstd*mean(dots)) and dgamma partials over xhat."""
    t = acc * w
    xhat = y * rstd
    return {"D": t * rstd + c, "dots": (t, xhat), "dw": (acc, xhat)}


# --- Variant mod factories ----------------------------------------------------
# The gemm_interface entry points (gemm_act / gemm_dact / gemm_norm_act /
# gemm_rms) ride these. The operand names (mAuxOut, mRowVecBroadcast,
# mColVecBroadcast, mColVecReduce, sr_seed) are the wire names shared with
# run_gemm_epi_plan epi_values and concat_layout keys.
#
# The frontend derives a fn's operands from its SIGNATURE, and an absent
# operand must not exist in the signature at all (that is what compiles the
# term out), so each present-operand combination is its own generated fn:
# the factory assembles the source, exec-compiles it, and the fail-closed
# semantic fingerprint keys on the code object (per-combination bytecode +
# the referenced activation fn's own source through getclosurevars).
#
# Why generated source and not a closure over the flags: a closure would be
# ONE function whose parameter list carries every possible operand — but the
# signature IS the frontend's interface. A dead `bias_n` parameter would be
# inferred as a real RowVecLoad operand (smem, cp.async, a loop input the
# vectorizer must chew), and passing a neutral value instead (bias=0.0)
# reaches the kernel as a live runtime term, not a compiled-out one. Faking
# it via __signature__ doesn't help either: operand kinds, the trace-time
# call, and the co_varnames-based fingerprint all read the real code object.
# Generating the def is the only path where "operand absent" means "term
# does not exist in the kernel".


def _gen_epi_fn(fname, tag, params, body, ns):
    lines = [f"def {fname}({', '.join(['acc', *params])}):"]
    lines.append(f'    """generated epilogue [{tag}]"""')
    lines += [f"    {ln}" for ln in body]
    src = "\n".join(lines)
    code = compile(src, f"<quack-epilogue:{tag}>", "exec")
    ns = {**ns, "unpack": unpack, "pack": pack, "__name__": "torch._vendor.quack.epilogue.library_generated"}
    exec(code, ns)
    return ns[fname]


def _vec_pins(params):
    pins = {}
    if "mRowVecBroadcast" in params:
        pins["mRowVecBroadcast"] = RowVecLoad("mRowVecBroadcast")
    if "mColVecBroadcast" in params:
        pins["mColVecBroadcast"] = ColVecLoad("mColVecBroadcast")
    return pins


_SR_OPS = (Scalar("sr_seed", dtype=Int32),)


@functools.lru_cache(maxsize=None)
def linear_act_mod(activation, *, gated, has_c, has_rowvec, has_colvec, sr=False, has_alpha=False):
    """gemm_act/gemm_gated as a mod: D = alpha * acc (+ C + rowvec + colvec),
    aux = act(D). Math order matches apply_linear_epilogue: alpha scales the
    accumulator only (before C / rowvec / colvec), so ``act(alpha * A @ B)``
    is a pre-activation scale."""
    fn_map = gate_fn_map if gated else act_fn_map
    act = fn_map[activation]
    params, body = [], []
    if has_alpha:
        params.append("alpha")
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    if has_colvec:
        params.append("mColVecBroadcast")
    expr = "acc"
    if has_alpha:
        body.append("x = acc * alpha")
        expr = "x"
    if has_c:
        body.append(f"x = {expr} + c")
        expr = "x"
    if has_rowvec:
        body.append(f"x = {expr} + mRowVecBroadcast")
        expr = "x"
    if has_colvec:
        body.append(f"x = {expr} + mColVecBroadcast")
        expr = "x"
    if gated:
        body.append(f"g, u = unpack({expr})")
        body.append('return {"D": pack(g, u), "mAuxOut": act(g, u)}')
    elif act is not None:
        body.append(f'return {{"D": {expr}, "mAuxOut": act({expr})}}')
    else:
        body.append(f'return {{"D": {expr}, "mAuxOut": {expr}}}')
    tag = (
        f"act:{activation}:g{int(gated)}c{int(has_c)}r{int(has_rowvec)}"
        f"v{int(has_colvec)}a{int(has_alpha)}"
    )
    fn = _gen_epi_fn("linear_act_epi", tag, params, body, {"act": act})
    ops = _vec_pins(params)
    if has_alpha:
        ops["alpha"] = Scalar("alpha")
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=ops,
        mode="acc_pair" if gated else None,
        extra_ops=_SR_OPS if sr else (),
    )(fn)


@functools.lru_cache(maxsize=None)
def norm_act_mod(activation, *, gated, has_c, has_rowvec, has_colvec, sr=False, has_alpha=False):
    """gemm_norm_act as a mod: x = (alpha * acc + C) * colvec * rowvec; D = x,
    aux = act(x). Scale order: colvec then rowvec. ``alpha`` scales the
    accumulator ONLY (before C and the norm scales) — it exists to carry NVFP4
    per-tensor dequant scales, which belong to the matmul product alone."""
    fn_map = gate_fn_map if gated else act_fn_map
    act = fn_map[activation]
    params, body = [], []
    if has_alpha:
        params.append("alpha")
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    if has_colvec:
        params.append("mColVecBroadcast")
    expr = "acc"
    if has_alpha:
        body.append("x = acc * alpha")
        expr = "x"
    if has_c:
        body.append(f"x = {expr} + c")
        expr = "x"
    if has_colvec:
        body.append(f"x = {expr} * mColVecBroadcast")
        expr = "x"
    if has_rowvec:
        body.append(f"x = {expr} * mRowVecBroadcast")
        expr = "x"
    if gated:
        body.append(f"g, u = unpack({expr})")
        body.append('return {"D": pack(g, u), "mAuxOut": act(g, u)}')
    elif act is not None:
        body.append(f'return {{"D": {expr}, "mAuxOut": act({expr})}}')
    else:
        body.append(f'return {{"D": {expr}, "mAuxOut": {expr}}}')
    tag = (
        f"norm_act:{activation}:g{int(gated)}c{int(has_c)}r{int(has_rowvec)}"
        f"v{int(has_colvec)}a{int(has_alpha)}"
    )
    fn = _gen_epi_fn("norm_act_epi", tag, params, body, {"act": act})
    ops = _vec_pins(params)
    if has_alpha:
        ops["alpha"] = Scalar("alpha")
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=ops,
        mode="acc_pair" if gated else None,
        extra_ops=_SR_OPS if sr else (),
    )(fn)


@functools.lru_cache(maxsize=None)
def dact_mod(activation, *, has_scale=False, has_reduce=False):
    """gemm_dact as a mod: c is the preact, acc is dout; D = dx, aux = act(c).
    Scale multiplies dout (dact is linear in it) and the postact; reduce
    accumulates postact * unscaled dout, matching dgated_mod."""
    dact = dact_fn_map[activation]
    params = ["c", "mColVecBroadcast"] if has_scale else ["c"]
    dout = "acc * mColVecBroadcast" if has_scale else "acc"
    if dact is None:
        body, dx, out = [], dout, "c"
    else:
        body, dx, out = [f"dx, out = dact(c, {dout})"], "dx", "out"
    postact = f"{out} * mColVecBroadcast" if has_scale else out
    if has_reduce:
        body.append(f'return {{"D": {dx}, "mAuxOut": {postact}, "mColVecReduce": ({out}, acc)}}')
    else:
        body.append(f'return {{"D": {dx}, "mAuxOut": {postact}}}')
    tag = f"dact:{activation}:s{int(has_scale)}r{int(has_reduce)}"
    fn = _gen_epi_fn("dact_epi", tag, params, body, {"dact": dact})
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=_vec_pins(params),
        reduces={"mColVecReduce": ColVecReduce("mColVecReduce", scaled=True)}
        if has_reduce
        else None,
    )(fn)


@functools.lru_cache(maxsize=None)
def dgated_mod(activation, *, has_scale, has_reduce):
    """gemm_dgated as a mod: acc = dout (per pair), c = packed (x, y) preact.
    Reduce accumulates postact * unscaled dout; postact is scaled after."""
    dgate = dgate_fn_map[activation]
    params, body = ["c"], []
    if has_scale:
        params.append("mColVecBroadcast")
    body.append("x, y = unpack(c)")
    dout = "acc * mColVecBroadcast" if has_scale else "acc"
    body.append(f"dx, dy, out = dgate(x, y, {dout})")
    postact = "out * mColVecBroadcast" if has_scale else "out"
    if has_reduce:
        # Scaled reduce: return the factors so the fold is one
        # fma(out, dout, acc) per pair.
        body.append(
            f'return {{"D": pack(dx, dy), "mAuxOut": {postact}, "mColVecReduce": (out, acc)}}'
        )
    else:
        body.append(f'return {{"D": pack(dx, dy), "mAuxOut": {postact}}}')
    tag = f"dgated:{activation}:s{int(has_scale)}r{int(has_reduce)}"
    fn = _gen_epi_fn("dgated_epi", tag, params, body, {"dgate": dgate})
    return gemm_epilogue(
        outputs=("mAuxOut",),
        ops=_vec_pins(params),
        reduces={"mColVecReduce": ColVecReduce("mColVecReduce", scaled=True)}
        if has_reduce
        else None,
        mode="packed_cd_b16x2",
    )(fn)


@functools.lru_cache(maxsize=None)
def rstd_gated_mod(activation):
    """GEMM2 of a block: apply the deferred rstd (colvec), then gated-activation
    pairs (gate_fn_map key: swiglu = llama, geglu = Gemma)."""
    gate = gate_fn_map[activation]
    body = [
        "g, u = unpack(acc * rstd)",
        'return {"postact": gate(g, u)}',
    ]
    fn = _gen_epi_fn("rstd_gated_epi", f"rstd_gated:{activation}", ["rstd"], body, {"gate": gate})
    return gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")}, mode="acc_pair")(
        fn
    )


@functools.lru_cache(maxsize=None)
def rstd_gated_preact_mod(activation):
    """rstd_gated_mod that ALSO stores the UNSCALED preact (D = the raw
    accumulator pairs) — the training-mode gate_up epilogue: the saved preact
    is the exact operand dgated_rstd_preact_mod's backward needs (rstd is
    saved separately as an f32 colvec, so scaling stays exact in bwd)."""
    gate = gate_fn_map[activation]
    body = [
        "g, u = unpack(acc)",
        "gs, us = unpack(acc * rstd)",
        'return {"D": pack(g, u), "postact": gate(gs, us)}',
    ]
    fn = _gen_epi_fn(
        "rstd_gated_preact_epi", f"rstd_gated_preact:{activation}", ["rstd"], body, {"gate": gate}
    )
    return gemm_epilogue(outputs=("postact",), ops={"rstd": ColVecLoad("rstd")}, mode="acc_pair")(
        fn
    )


@functools.lru_cache(maxsize=None)
def dgated_rstd_preact_mod(activation):
    """Backward of s = gate(rstd * GU) — scale BEFORE activation, the
    llama/Gemma rstd_gated_preact_mod convention (dgated_mod/dswiglu_norm_mod
    invert s = rstd * gate(GU), scale-after). acc = ds (dout), c = packed
    UNSCALED preact GU, rstd colvec. Emits D = dGU (grad wrt the unscaled
    preact — rstd folded here so both downstream bwd GEMMs are plain), the
    exact recomputed postact s (fc2-wgrad operand), and the rstd-gradient stat
    dsum = sum_pairs(dgs*g + dus*u) per tile (host: corr = -(rstd^3/d)*sum).
    dsum is a plain reduce: the stat is a two-product sum, so the scaled
    (val, scale) fma fold does not apply."""
    dgate = dgate_fn_map[activation]
    body = [
        "g, u = unpack(c)",
        "gs, us = g * rstd, u * rstd",
        "dgs, dus, s = dgate(gs, us, acc)",
        'return {"D": pack(dgs * rstd, dus * rstd), "postact": s, "dsum": dgs * g + dus * u}',
    ]
    fn = _gen_epi_fn(
        "dgated_rstd_preact_epi",
        f"dgated_rstd_preact:{activation}",
        ["c", "rstd"],
        body,
        {"dgate": dgate},
    )
    return gemm_epilogue(
        outputs=("postact",),
        ops={"rstd": ColVecLoad("rstd")},
        reduces={"dsum": ColVecReduce("dsum")},
        mode="packed_cd_b16x2",
    )(fn)


rstd_swiglu_epi = rstd_gated_mod("swiglu")
rstd_swiglu_preact_epi = rstd_gated_preact_mod("swiglu")
dswiglu_rstd_preact_mod = dgated_rstd_preact_mod("swiglu")


@functools.lru_cache(maxsize=None)
def sq_reduce_mod(*, has_c, has_rowvec, has_aux, has_alpha=False):
    """gemm_rms's sq-reduce as a mod: x = alpha * acc (+ C); reduce[m] +=
    sum_n x^2 (before the rowvec scale); optional aux = x; D = x * rowvec.
    ``alpha`` scales the accumulator ONLY (NVFP4 per-tensor dequant scales);
    it MUST land before the sq-reduce or the host rsqrt normalizes the wrong
    magnitude (RMSNorm's scale invariance would hide most of the error)."""
    params, body = [], []
    if has_alpha:
        params.append("alpha")
    if has_c:
        params.append("c")
    if has_rowvec:
        params.append("mRowVecBroadcast")
    expr = "acc"
    if has_alpha:
        body.append("x = acc * alpha")
        expr = "x"
    if has_c:
        body.append(f"x = {expr} + c")
        expr = "x"
    d = f"{expr} * mRowVecBroadcast" if has_rowvec else expr
    # Scaled reduce: (x, x) folds as one fma(x, x, acc) per pair instead of
    # FMUL+FADD.
    ret = f'"D": {d}, "mColVecReduce": ({expr}, {expr})'
    if has_aux:
        ret += f', "mAuxOut": {expr}'
    body.append(f"return {{{ret}}}")
    tag = f"sq_reduce:c{int(has_c)}r{int(has_rowvec)}a{int(has_aux)}s{int(has_alpha)}"
    fn = _gen_epi_fn("sq_reduce_epi", tag, params, body, {})
    # No host finalize: __call__ returns the RAW per-tile partials, and
    # gemm_rms fuses sum + rsqrt in rms_final_reduce (bitwise-equal to the
    # pre-object pipeline; a host torch.sum would reorder the fold).
    reduce = ColVecReduce("mColVecReduce", scaled=True)
    reduce.host_finalize = None
    ops = _vec_pins(params)
    if has_alpha:
        ops["alpha"] = Scalar("alpha")
    return gemm_epilogue(
        outputs=("mAuxOut",) if has_aux else (),
        ops=ops,
        reduces={"mColVecReduce": reduce},
    )(fn)


# --- SigLIP / deferred-LayerNorm transformer block ----------------------------
# LayerNorm defers through a GEMM like RMSNorm with ONE extra colvec: for
#   z = LN(h) @ W^T + b,  LN(h) = (h - mu)*sig*gamma + beta, sig = rsqrt(var+eps)
#   z = sig ⊙ [(h*gamma) @ W^T] - (sig*mu) ⊙ (W@gamma) + (W@beta + b)
# The producing GEMM writes a = h*gamma plus (sum, sqsum) partials (host
# finalizes mu, sig); the consuming GEMM applies two colvecs (s = sig,
# t = sig*mu) and two HOST-PRECOMPUTED per-layer rowvecs (wg = W@gamma,
# wb = W@beta + linear_bias). Backward: the boundary's stat reduces (r1 =
# sum_j dz*z, column-sums dwb/dwg for dbeta and the rank-1 dW corrections)
# live in the epilogue that PRODUCES dz — one GEMM earlier, so the correction
# colvecs arrive finalized at the next dgrad's apply epilogue (no self-stat).
# Full algebra + host finalizers: tests/test_siglip_pipeline.py; procedure:
# skills/codaify.


@gemm_epilogue(
    ops={
        "s": ColVecLoad("s"),
        "t": ColVecLoad("t"),
        "wg": RowVecLoad("wg"),
        "wb": RowVecLoad("wb"),
    }
)
def ln_affine_epi(acc, s, t, wg, wb):
    """Consuming GEMM of a deferred-LayerNorm boundary (SigLIP qkv proj):
    D = s*acc - t*wg + wb."""
    return {"D": (acc * s - t * wg) + wb}


@functools.lru_cache(maxsize=None)
def ln_affine_act_mod(activation):
    """SigLIP/ViT fc1: deferred-LN affine then activation (act_fn_map key —
    gelu_tanh_approx for SigLIP, gelu_erf for timm/torchvision ViT, quick_gelu
    for CLIP); D = the bf16 preact (saved for dact in bwd), postact feeds fc2."""
    act = act_fn_map[activation]
    body = [
        "z = (acc * s - t * wg) + wb",
        'return {"D": z, "postact": act(z)}',
    ]
    fn = _gen_epi_fn(
        "ln_affine_act_epi",
        f"ln_affine_act:{activation}",
        ["s", "t", "wg", "wb"],
        body,
        {"act": act},
    )
    return gemm_epilogue(
        outputs=("postact",),
        ops={
            "s": ColVecLoad("s"),
            "t": ColVecLoad("t"),
            "wg": RowVecLoad("wg"),
            "wb": RowVecLoad("wb"),
        },
    )(fn)


ln_affine_gelu_epi = ln_affine_act_mod("gelu_tanh_approx")


@gemm_epilogue(
    outputs=("resid_out",),
    ops={"bias": RowVecLoad("bias"), "weight": RowVecLoad("weight")},
    reduces={
        "hsum": ColVecReduce("hsum"),
        "sqsum": ColVecReduce("sqsum", scaled=True),
    },
)
def ln_partial_epi(acc, c, bias, weight):
    """Producing GEMM of a deferred-LayerNorm boundary (SigLIP out_proj/fc2):
    y = acc + linear bias + residual(C); write the residual stream, the
    gamma-applied next-GEMM input, and BOTH LayerNorm stats' per-tile partials
    (host: mu = sum/d, sig = rsqrt(sqsum/d - mu^2 + eps), t = sig*mu)."""
    y = acc + bias + c
    return {"D": y * weight, "resid_out": y, "hsum": y, "sqsum": (y, y)}


@functools.lru_cache(maxsize=None)
def dact_ln_stats_mod(activation, sinks="full"):
    """SigLIP/ViT fc2-dgrad: acc = dpostact, c = saved bf16 preact z. dz
    through act' (dact_fn_map key); D = s*dz (the boundary's sig folded so
    fc1-dgrad AND fc1-wgrad stay plain). Boundary-bwd stats: r1 = per-row sum
    dz*z (feeds dsig with the wb-dot host correction), dwb = column-sum dz
    (dbeta + linear-bias grad + rank-1 dW term), dwg = column-sum t*dz
    (dgamma's W-path + rank-1 dW term).

    ``sinks`` trades in-kernel rowvec sinks for host GEMVs off the stored D
    (iket: the VecReduce end-loop flush is ~60% of epilogue time at K~1152;
    every colsum is recoverable because D = s*dz is bijective per row):
      * "full": r1 + dwb + dwg in-kernel (the original).
      * "dwb":  r1 + dwb; host dwg = mu^T @ D with mu = t/s (one GEMV).
      * "r1":   r1 only;  host dwb = (1/s)^T @ D, dwg = mu^T @ D — one
        (2, m) @ (m, n) GEMM re-reading D once.
    The dropped ``t`` colvec leaves the signature entirely (compiled out)."""
    dact = dact_fn_map[activation]
    ret = {
        "full": 'return {"D": dz * s, "r1": (dz, c), "dwb": dz, "dwg": (dz, t)}',
        "dwb": 'return {"D": dz * s, "r1": (dz, c), "dwb": dz}',
        "r1": 'return {"D": dz * s, "r1": (dz, c)}',
    }[sinks]
    params = ["c", "s", "t"] if sinks == "full" else ["c", "s"]
    body = ["dz, _ = dact(c, acc)", ret]
    fn = _gen_epi_fn(
        "dact_ln_stats_epi", f"dact_ln_stats:{activation}:{sinks}", params, body, {"dact": dact}
    )
    ops = {"s": ColVecLoad("s")}
    if sinks == "full":
        ops["t"] = ColVecLoad("t")
    reduces = {"r1": ColVecReduce("r1", scaled=True)}
    if sinks in ("full", "dwb"):
        reduces["dwb"] = RowVecReduce("dwb")
    if sinks == "full":
        reduces["dwg"] = RowVecReduce("dwg", scaled=True)
    return gemm_epilogue(ops=ops, reduces=reduces)(fn)


dgelu_ln_stats_epi = dact_ln_stats_mod("gelu_tanh_approx")


@functools.lru_cache(maxsize=None)
def ln_bwd_apply_mod(sinks="full"):
    """LayerNorm-bwd apply around a dgrad GEMM (SigLIP fc1-dgrad/qkv-dgrad):
    acc = da (grad of a = h*gamma), c = residual grad, y = saved residual h.
    D = dh_total = acc*w + c + y*corr_mul + corr_add, with the two correction
    colvecs host-finalized from the PREVIOUS bwd GEMM's boundary stats:
      dsig = (r1 - sum_j dz*wb)/sig ; dmu = -sig * sum_j dz*wg
      corr_mul = -dsig*sig^3/d ; corr_add = (dmu + dsig*sig^3*mu)/d.
    dw partials = dgamma's a-path (sum_m da*h); dbias = column-sums of the
    OUTPUT dh (the producing linear's bias grad lives downstream).

    ``sinks``: "full" = dw + dbias in-kernel; "dw" drops the dbias
    RowVecReduce — dbias is the column-sum of the STORED D, recoverable
    host-side (ones GEMV / fold into a downstream read of D). dw = sum_m
    acc*y is NOT recoverable (acc unsaved)."""
    body = [
        "dh = acc * w + c + y * corr_mul + corr_add",
        'return {"D": dh, "dw": (acc, y), "dbias": dh}'
        if sinks == "full"
        else 'return {"D": dh, "dw": (acc, y)}',
    ]
    fn = _gen_epi_fn(
        "ln_bwd_apply_gen_epi",
        f"ln_bwd_apply:{sinks}",
        ["c", "y", "w", "corr_mul", "corr_add"],
        body,
        {},
    )
    reduces = {"dw": RowVecReduce("dw", scaled=True)}
    if sinks == "full":
        reduces["dbias"] = RowVecReduce("dbias")
    return gemm_epilogue(
        ops={
            "w": RowVecLoad("w"),
            "corr_mul": ColVecLoad("corr_mul"),
            "corr_add": ColVecLoad("corr_add"),
        },
        reduces=reduces,
    )(fn)


ln_bwd_apply_epi = ln_bwd_apply_mod("full")


@functools.lru_cache(maxsize=None)
def dact_dbias_mod(activation):
    """Standard-organization fc2-dgrad (SigLIP/ViT): act' (dact_fn_map key)
    from the saved preact c, UNSCALED dz out (the boundary's LN-bwd runs as a
    separate narrow fused kernel), plus the fc1-bias grad column-sums. Contrast
    dact_ln_stats_mod (deferred org: sig-folded D + 3 boundary-stat sinks)."""
    dact = dact_fn_map[activation]
    body = [
        "dz, _ = dact(c, acc)",
        'return {"D": dz, "dwb": dz}',
    ]
    fn = _gen_epi_fn("dact_dbias_epi", f"dact_dbias:{activation}", ["c"], body, {"dact": dact})
    return gemm_epilogue(reduces={"dwb": RowVecReduce("dwb")})(fn)


dgelu_dbias_epi = dact_dbias_mod("gelu_tanh_approx")
