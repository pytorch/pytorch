# Copyright (c) 2026, Han Guo, Tri Dao.
"""Rotary-position resources and ready-to-use RoPE GEMM epilogues."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr

import torch._vendor.quack.copy_utils as copy_utils
from torch._vendor.quack.epilogue.ops import ColVecLoad, EpiOp, RowVecLoad, TileLoad
from torch._vendor.quack.epilogue.frontend import gemm_epilogue, pack, unpack


class RotaryCosSinLoad(EpiOp):
    """Per-subtile gmem->rmem load of the interleaved rotary cos/sin table.

    The param tensor is (seqlen_ro, head_dim), row-major, cos at even columns
    and sin at odd columns. Output column n reads table column n % head_dim of
    table row m: the head broadcast is expressed as a stride-0 repeat mode in
    the per-tile gmem view, so a single ``partition_for_epilogue`` makes the
    loaded fragment elementwise-aligned with tRS_rD.

    Requires tile_N % head_dim == 0 or head_dim % tile_N == 0 (the repeat /
    slice views below are built with static layout algebra; a tile that
    straddles a head boundary at a non-multiple offset would need a per-element
    mod). Rows beyond the M limit are left at 0 — those lanes never reach gmem
    (the D store is bound-checked) but must not fault.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # Stronger than assume_stride_divisibility (32 bits): assume 16-byte
        # stride divisibility so gmem loads can vectorize to 128 bits. The host
        # asserts head_dim (== the row stride of a contiguous table) allows it.
        tensor = getattr(args, self.name)
        divby = 128 // tensor.element_type.width
        new_stride = tuple(
            cute.assume(s, divby=divby) if not cute.is_static(s) else s for s in tensor.stride
        )
        return {
            self.name: cute.make_tensor(
                tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride)
            )
        }

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        head_dim = const_expr(param.shape[1])
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        assert tile_N % head_dim == 0 or head_dim % tile_N == 0, (
            "rotary epilogue requires tile_N to be a multiple or a divisor of head_dim"
        )
        m_tile = ctx.tile_coord_mnkl[0]
        mCS = param
        if const_expr(ctx.varlen_manager.varlen_m):
            mCS = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx], 0), mCS
            )
        if const_expr(tile_N % head_dim == 0):
            # One tile covers >= 1 whole head: repeat the head_dim columns via a
            # stride-0 mode. The view is identical for every N tile coordinate.
            gCS_rows = cute.local_tile(mCS, (tile_M, head_dim), (m_tile, 0))
            gCS = cute.make_tensor(
                gCS_rows.iterator,
                cute.make_layout(
                    (tile_M, (head_dim, tile_N // head_dim)),
                    stride=(gCS_rows.stride[0], (gCS_rows.stride[1], 0)),
                ),
            )
        else:
            # A head spans several tiles: slice the head's columns for this tile.
            gCS = cute.local_tile(
                mCS,
                (tile_M, tile_N),
                (m_tile, ctx.tile_coord_mnkl[1] % (head_dim // tile_N)),
            )
        tDgCS = ctx.partition_for_epilogue_fn(gCS)
        tDcCS = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDgCS = ctx.tiled_copy_r2s.retile(tDgCS)
            tDcCS = ctx.tiled_copy_r2s.retile(tDcCS)
        tDgCS = cute.group_modes(tDgCS, 3, cute.rank(tDgCS))
        tDcCS = cute.group_modes(tDcCS, 3, cute.rank(tDcCS))
        limit_m = min(ctx.varlen_manager.len_m(ctx.batch_idx) - m_tile * tile_M, tile_M)
        full_tile = Boolean(limit_m >= tile_M)
        # Subtile iteration count decides the prefetch depth below.
        num_subtiles_static = const_expr(
            cute.size(
                cute.zipped_divide(
                    cute.make_layout(gemm.cta_tile_shape_mnk[:2]), ctx.epi_tile
                ).shape[1]
            )
        )
        # Two per-subtile fragments (double buffer): subtile i+1's loads are
        # issued in begin_loop(i) so their latency hides behind subtile i's
        # rotation + store. With a single subtile per tile (QK-norm's wide epi
        # tiles), skip the second buffer — it would only add register pressure.
        # Layout-matched to the gmem partition; integer indexing is
        # coordinate-order, so they still line up elementwise with tRS_rD.
        # Rows beyond limit_m keep the initial 0.
        buf0 = cute.make_rmem_tensor_like(tDgCS[None, None, None, 0])
        buf0.fill(0.0)
        if const_expr(num_subtiles_static > 1):
            buf1 = cute.make_rmem_tensor_like(tDgCS[None, None, None, 0])
            buf1.fill(0.0)
        else:
            buf1 = buf0
        bufs = [buf0, buf1]
        # Explicit vector width for the gmem->rmem copy: autovec is too
        # conservative for gmem sources, so we build the copy atom ourselves.
        src0_f = cute.coalesce(cute.filter_zeros(tDgCS[None, None, None, 0]))
        dst0_f = cute.coalesce(cute.filter_zeros(bufs[0]))
        copy_vec = const_expr(
            min(cute.max_common_vector(src0_f, dst0_f), 128 // param.element_type.width)
        )
        assert head_dim % copy_vec == 0, "head_dim must be a multiple of the copy vector"
        # Subtile iteration order, to know which epi_coord comes next. Must
        # match the epi_tile_layout built in GemmBase.epilogue.
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), ctx.epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(gemm.epi_m_major) else (1, 0)
        )
        if const_expr(param.element_type != gemm.acc_dtype):
            tDrCS_cvt = cute.make_rmem_tensor(bufs[0].shape, gemm.acc_dtype)
        else:
            tDrCS_cvt = None
        state = [tDgCS, tDcCS, full_tile, limit_m, bufs, tDrCS_cvt, copy_vec, epi_tile_layout]
        # Preload subtile 0 so begin_loop(0) finds it ready.
        self._load_subtile(state, epi_tile_layout.get_hier_coord(0), 0)
        return state

    @cute.jit
    def _load_subtile(self, state, epi_coord, buf_idx: cutlass.Constexpr[int]):
        tDgCS, tDcCS, full_tile, limit_m, bufs, _, copy_vec, _ = state
        tDgCS_cur = tDgCS[None, None, None, epi_coord]
        buf = bufs[buf_idx]
        if full_tile:
            tiler = cute.make_layout(copy_vec)
            copy_atom = copy_utils.get_copy_atom(buf.element_type, copy_vec)
            cute.copy(
                copy_atom,
                cute.zipped_divide(cute.coalesce(cute.filter_zeros(tDgCS_cur)), tiler),
                cute.zipped_divide(cute.coalesce(cute.filter_zeros(buf)), tiler),
            )
        else:
            # Ragged last M tile: per-element row-predicated loads. Slow but
            # only ever runs on the boundary tile.
            tDcCS_cur = tDcCS[None, None, None, epi_coord]
            for i in cutlass.range(cute.size(buf), unroll_full=True):
                if tDcCS_cur[i][0] < limit_m:
                    buf[i] = tDgCS_cur[i]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        bufs, tDrCS_cvt, epi_tile_layout = state[4], state[5], state[7]
        idx = const_expr(cute.crd2idx(epi_coord, epi_tile_layout))
        num_subtiles = const_expr(cute.size(epi_tile_layout))
        if const_expr(idx + 1 < num_subtiles):
            self._load_subtile(state, epi_tile_layout.get_hier_coord(idx + 1), (idx + 1) % 2)
        cur = bufs[idx % 2]
        if const_expr(tDrCS_cvt is not None):
            tDrCS_cvt.store(cur.load().to(tDrCS_cvt.element_type))
            return tDrCS_cvt
        return cur


class RotaryCosSinLoadHost(RotaryCosSinLoad):
    """The epirope op + the generic host-layer schema hooks it predates."""

    fn_port = "value"

    def host_arg_key(self, value):
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map

        # head_dim must be static in the fake (begin() does layout algebra on it).
        return (torch2cute_dtype_map[value.dtype], value.shape[-1])

    def host_fake_arg(self, key, fctx):
        from torch._vendor.quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(
            dtype, (cute.sym_int(), head_dim), leading_dim=1, divisibility=128 // dtype.width
        )


class RotaryCosSinTMALoad(TileLoad):
    """The (seqlen_ro, head_dim) interleaved cos/sin table staged through the
    TMA epilogue load pipeline instead of per-tile gmem->rmem copies
    (RotaryCosSinLoad). Default choice — see ``rotary_cos_sin_load``.

    TMA descriptors cannot encode the stride-0 head broadcast, so the wrap
    lives in the copy COORDINATES: each epi subtile TMA-loads the table box
    at column ``output_col % head_dim`` (redundant loads across heads hit L2).
    The smem stage then holds exactly the subtile-aligned cos/sin slice, so
    the whole TileLoad consumer path (S2R, staging, pipeline tx accounting)
    is inherited unchanged. Dense-only for now (TMA loads have no varlen_m
    ragged-descriptor path; use the LDG op under varlen).
    """

    def __init__(self, name):
        super().__init__(name)

    def host_arg_key(self, value):
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map

        # head_dim static in the fake: the copy fn branches on it at trace time.
        return (torch2cute_dtype_map[value.dtype], value.shape[-1])

    def host_fake_arg(self, key, fctx):
        from torch._vendor.quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(
            dtype, (cute.sym_int(), head_dim), leading_dim=1, divisibility=128 // dtype.width
        )

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        assert not varlen_manager.varlen_m, (
            "RotaryCosSinTMALoad does not support varlen_m; use "
            "rotary_cos_sin_load(name, tma=False) (rope_table_ldg_epi / qk_rope_ldg_epi)"
        )
        # The TMA prep appends a stride-0 batch mode; slice it off (all
        # batches share the table).
        tensor = varlen_manager.offset_batch_epi(getattr(params, self.name), tile_coord_mnkl[3])
        head_dim = const_expr(tensor.shape[1])
        tile_M, tile_N = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        epi_tile = getattr(params, self._epi_tile_key())
        epi_N = const_expr(cute.size(epi_tile[1]))
        atom = getattr(params, self._tma_atom_key())
        if const_expr(tile_N % head_dim == 0):
            # Tile covers >= 1 whole head: tile the table (tile_M, head_dim)
            # and wrap the epi-subtile N coordinate modulo subtiles-per-head.
            assert head_dim % epi_N == 0, "head_dim must be a multiple of the epi tile N"
            subtiles_per_head = const_expr(head_dim // epi_N)
            copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
                atom,
                tensor,
                (tile_M, head_dim),
                epi_tile,
                smem_tensor,
                (tile_coord_mnkl[0], 0),
            )
            inner = copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)
            # Callers pass either the hier (epi_m, epi_n) coord (gemm_base's
            # inline path) or a linear subtile index (the SM100 dedicated
            # epi-load warp): decode linear via the same ordered layout the
            # store loop uses so pipeline stage order matches consumption.
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape, order=(0, 1) if const_expr(gemm.epi_m_major) else (1, 0)
            )

            def copy_fn(src_idx, producer_state, **kw):
                coord = (
                    src_idx
                    if isinstance(src_idx, tuple)
                    else epi_tile_layout.get_hier_coord(src_idx)
                )
                inner((coord[0], coord[1] % subtiles_per_head), producer_state, **kw)

            return copy_fn
        # A head spans several tiles: pick the head-relative N tile, no
        # per-subtile wrap needed.
        assert head_dim % tile_N == 0, (
            "rotary epilogue requires tile_N to be a multiple or a divisor of head_dim"
        )
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            atom,
            tensor,
            (tile_M, tile_N),
            epi_tile,
            smem_tensor,
            (tile_coord_mnkl[0], tile_coord_mnkl[1] % (head_dim // tile_N)),
        )
        return copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)


def rotary_cos_sin_load(name, tma=True):
    """Rotary cos/sin table op. ``tma=True`` (default) stages the table via
    the TMA epilogue load pipeline; ``tma=False`` is the per-tile gmem->rmem
    op — required for varlen_m and pre-TMA archs (SM80), and 1-3% faster at
    large non-pingpong tiles.

    Why TMA is the default — H100, m=16384 k=4096 head_dim=128, rope overhead
    vs an identity epilogue at the same config (interleaved-median bench):

    ==================  =======  =======
    config              LDG      TMA
    ==================  =======  =======
    256x128 c(1,2)      +15.3%   +15.5%   best PLAIN-GEMM config
    128x256 c(1,1)      +13.6%   +10.0%
    128x128 c(1,2) pp    +6.2%    +1.7%
    192x128 c(1,2) pp    +4.3%    +1.3%   best rope/qknorm config
    ==================  =======  =======

    Under pingpong the producer-warp TMA staging escapes the exclusive
    per-warpgroup epilogue window that serializes consumer LDGs; at 192-row
    pingpong tiles the LDG double-buffer register cost tips composed
    epilogues into spills (qk_rope: LDG +23% vs TMA +2.5%). Clustered
    pingpong + TMA is the absolute winner for the fused QK-norm/RoPE
    epilogues (qk_rope 794us vs 918us at the best non-pingpong config), so
    the default optimizes the configs these epilogues actually run on.
    """
    return RotaryCosSinTMALoad(name) if tma else RotaryCosSinLoadHost(name)


@gemm_epilogue(ops={"cs": rotary_cos_sin_load("cs")}, mode="acc_pair")
def rope_table_epi(acc, cs, bias):
    """RoPE from the real (seqlen, head_dim) table op, composed with a rowvec
    bias in fn math — the op is a value port; rotation order is explicit."""
    x1, x2 = unpack(acc + bias)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(ops={"cs": rotary_cos_sin_load("cs", tma=False)}, mode="acc_pair")
def rope_table_ldg_epi(acc, cs, bias):
    """rope_table_epi on the gmem->rmem table op: required for varlen_m
    (table indexed by global flattened row), pre-TMA archs, and marginally
    faster at large non-pingpong tiles (see rotary_cos_sin_load)."""
    x1, x2 = unpack(acc + bias)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(ops={"pos": ColVecLoad("pos"), "freq": RowVecLoad("freq")}, mode="acc_pair")
def rope_posfreq_epi(acc, pos, freq):
    """RoPE computing cos/sin in-kernel via ``sincos(pos * inv_freq)`` instead
    of loading the precomputed (seqlen_ro, head_dim) table.

    ``pos`` is the per-row position, (l, m) float32 (exact up to 2^24; may
    differ per batch, unlike the shared table). ``freq`` is the per-output-
    column inv-freq as a float-float table, (l, n) float32, hi at even
    columns and lo at odd (same interleaving convention as the cos/sin
    table) — build it with ``make_interleaved_inv_freq``. Both are plain
    broadcast vector loads, so the per-tile gmem traffic is tile_M + tile_N
    elements vs the table's tile_M * head_dim; the table's tile_N/head_dim
    alignment constraint disappears (the head wrap is baked into the freq
    table), and varlen_m works for free (rank-1 (total_m,) ``pos``).
    Per-column behavior is data, not kernel structure: zero-frequency columns
    rotate by angle 0, an exact identity, so packed QKV projections rotate
    only the Q/K block via ``make_interleaved_inv_freq(inv_freq, qk_dim,
    v_dim)`` — no branch, V passes through bitwise.

    Angle math: the f32 product pos*freq loses absolute precision linearly in
    the angle (a plain-f32 pipeline — including a table built from f32 host
    angles — is off by 5e-3 of output scale at 128k positions, 0.67 at 16M),
    so the angle is computed as a float-float, in TURNS: the table stores
    inv_freq/2pi, the reduction is an exact mod-1 (H100's MUFU natively works
    in turns — ptxas prepends an FMUL.RZ by 1/2pi to every ``sin.approx``, so
    reducing mod 2pi in radians is the same work in the wrong unit; mod-1
    needs no Cody-Waite constant split at all). Max error vs an f64 reference
    is ~8e-7 of output scale at every position range through 2^24 — same as a
    table built from f64 host angles (checked: in-kernel libdevice sincosf
    gains nothing and costs +17-65% kernel time). Pipe budget per rotation
    pair: 15 FMA-pipe ops + 2 MUFU — the quarter-rate XU pipe (MUFU) is the
    epilogue's math wall, which is why the round uses the magic-bias adds
    instead of FRND (FRND also issues on XU; evicting it cut the exposed
    k=512 rope overhead by ~4-6pp, ncu-verified). H100 m=16384 k=4096 n=4096
    head_dim=128, overhead vs a bias-only epilogue at the same config
    (interleaved-median bench), against rope_table_epi's TMA table:

    ==================  ========  =======
    config              posfreq   table
    ==================  ========  =======
    192x128 c(1,2) pp     +0.8%    +2.0%
    128x256 c(1,1)        +3.8%    +4.8%
    256x128 c(1,2)        +4.2%   +10.8%
    ==================  ========  =======
    """
    # Reference math, per rotation pair (x1, x2) at row position `pos`:
    #     theta = pos * inv_freq
    #     D = (x1*cos(theta) - x2*sin(theta), x1*sin(theta) + x2*cos(theta))
    # The angle helpers below compute this with three transformations that
    # lose no accuracy (turns units, float-float compensation, magic-bias
    # rounding) — see _angle_turns / _sincos_turns for the derivations.
    x1, x2 = unpack(acc)
    s, c = _sincos_turns(*_angle_turns(pos, freq))
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(ops={"pos": ColVecLoad("pos"), "freq": RowVecLoad("freq")}, mode="acc_pair")
def rope_posfreq_bias_epi(acc, pos, freq, bias):
    """rope_posfreq_epi with a rowvec bias added before the rotation (biased
    QKV projections; a separate mod because an absent operand must be absent
    from the signature to compile the term out)."""
    x1, x2 = unpack(acc + bias)
    s, c = _sincos_turns(*_angle_turns(pos, freq))
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={"pos": ColVecLoad("pos"), "freq": RowVecLoad("freq"), "rstd": ColVecLoad("rstd")},
    mode="acc_pair",
)
def rstd_rope_posfreq_epi(acc, pos, freq, rstd):
    """Mid-stack QKV epilogue of a Coda-style transformer block (deferred-rstd
    boundary): scale rows by the boundary's rstd colvec (the producing GEMM
    wrote a = h*gamma with the rstd deferred here — a row scale commutes
    through the GEMM), then rotate. Rope applies to the Q/K columns and V
    passes through bitwise via zero-frequency columns of the interleaved freq
    table (see rope_posfreq_epi; the row scale commutes with the rotation, so
    scaling before unpack is exact)."""
    x1, x2 = unpack(acc * rstd)
    s, c = _sincos_turns(*_angle_turns(pos, freq))
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


def _angle_turns(pos, freq):
    """theta = pos * inv_freq as an unevaluated float-float sum (t, lo), in
    TURNS (the freq table stores inv_freq/2pi — see make_interleaved_inv_freq).

    The plain product pos*fh rounds away low bits worth whole rotations once
    pos*freq is large (~0.5 ulp = 1 rotation at pos ~ 2^24, freq ~ 1). The
    exact FFMA residual of the product plus the below-f32 freq bits (fl)
    carry them in the compensation term `lo`, re-added after range reduction
    when the angle is small again (_sincos_turns)."""
    fh, fl = unpack(freq)  # float-float inv_freq/2pi: fh + fl to ~2^-49 rel
    t = pos * fh  # angle in turns, rounded to f32
    terr = cute.math.fma(pos, fh, -t)  # bits the product rounded away (exact)
    lo = cute.math.fma(pos, fl, terr)  # + the angle bits below fh's precision
    return t, lo


def _sincos_turns(t, lo):
    """(sin, cos) of the float-float angle (t + lo) turns via MUFU.

    Turns make range reduction exact: theta mod 1 is a round and a subtract,
    no Cody-Waite constant splitting — and turns are the hardware's native
    unit (MUFU.SIN takes turns; ptxas prepends x*(1/2pi) to every radians
    sin.approx). The round uses the magic-bias add because MUFU.SIN/COS and
    FRND all issue on the quarter-rate XU pipe, the epilogue's math
    bottleneck (the 2 MUFU are its floor) — full-rate FMA-pipe ops beat FRND
    (B300 128x256 k=512 exposed overhead vs bias-only: +47.7% vs +49.5%)."""
    # Round-to-nearest-even via the magic bias: exact for |t| < 2^22 (here
    # t < 2^24/2pi), because adding 1.5*2^23 shifts the integer part onto the
    # mantissa boundary. t - q is then exact (both are multiples of ulp(t)).
    # The biasing add is written as a math-dialect fma (t*1.0 + M, bitwise
    # identical to t + M; ptxas strength-reduces it back to FADD) so the
    # SM100 loop vectorizer scalarizes it: written as arith `(t + M) - M`,
    # t gets two vectorized-arith consumers, the dataflow shape that
    # heap-corrupts the closed vectorizer pass (segfault/double-free/compile
    # hang; minimal repro `pack(x1 + t, x2 + t)`). Same reason the sin/cos
    # below are separate calls, not the fused two-result cute.math.sincos.
    q = cute.math.fma(t, 1.0, 12582912.0) - 12582912.0
    r = (t - q) + lo  # theta mod 1, plus the compensation term
    # sin.approx wants radians and ptxas rescales by 1/2pi for MUFU; the
    # cancelling multiply pair is PTX's toll — there is no turns-domain PTX.
    # Separate sin/cos calls, NOT cute.math.sincos: the SM100 loop vectorizer
    # (the vectorize=True pair loop in gemm_epilogue) segfaults the MLIR
    # compiler on the fused two-result op. Identical SASS either way — ptxas
    # shares the FMUL.RZ between the MUFU.SIN/MUFU.COS pair of one argument.
    r_rad = r * 6.283185307179586
    s = cute.math.sin(r_rad, fastmath=True)
    c = cute.math.cos(r_rad, fastmath=True)
    return s, c


@gemm_epilogue(ops={"pos": ColVecLoad("pos"), "freq": RowVecLoad("freq")}, mode="acc_pair")
def rope_posfreq_scaled_epi(acc, pos, freq, bias, scale):
    """rope_posfreq_epi with an attention-temperature factor on the rotated
    output: D = scale * rope(acc + bias). This is YaRN's mscale (DeepSeek-V2/
    V3, Qwen long-context) — the table-op form bakes it into the host cos/sin
    values, but with in-kernel sincos it must be a multiply (an angle change
    cannot express a magnitude change), so it costs one Scalar operand and
    two FMULs per pair. The YaRN frequency transform itself is pure data:
    feed the transformed inv_freq to make_interleaved_inv_freq."""
    x1, x2 = unpack(acc + bias)
    s, c = _sincos_turns(*_angle_turns(pos, freq))
    x1, x2 = x1 * scale, x2 * scale
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={
        "pos_t": ColVecLoad("pos_t"),
        "pos_h": ColVecLoad("pos_h"),
        "pos_w": ColVecLoad("pos_w"),
        "freq_t": RowVecLoad("freq_t"),
        "freq_h": RowVecLoad("freq_h"),
        "freq_w": RowVecLoad("freq_w"),
    },
    mode="acc_pair",
)
def mrope_posfreq_epi(acc, pos_t, pos_h, pos_w, freq_t, freq_h, freq_w, bias):
    """Multimodal 3D RoPE (Qwen2-VL mRoPE): head-dim sections rotate by
    different position axes — pair j uses pos_t, pos_h, or pos_w according to
    which section j falls in. The section SELECT is data, not kernel
    structure: each axis gets its own freq table, zeroed outside its section
    (make_mrope_inv_freq), so the angle is the three-term dot product

        theta = pos_t*f_t(j) + pos_h*f_h(j) + pos_w*f_w(j)

    with exactly one nonzero term per column — the sums are exact (adding
    zeros), so the float-float compensation of the live term survives intact.
    Costs two extra vector-load pairs and ~6 FMA-pipe ops per pair over
    rope_posfreq_epi; the XU (MUFU) cost is unchanged."""
    x1, x2 = unpack(acc + bias)
    t1, lo1 = _angle_turns(pos_t, freq_t)
    t2, lo2 = _angle_turns(pos_h, freq_h)
    t3, lo3 = _angle_turns(pos_w, freq_w)
    # The t sums are math-dialect fmas (bitwise == the adds), NOT arith `+`:
    # each t_i already feeds its Dekker negf, and a second vectorized-arith
    # consumer is the SM100 vectorizer's crash shape (see _sincos_turns).
    # The lo_i are math.fma results — their arith sums are single-consumer
    # and pack fine.
    t12 = cute.math.fma(t1, 1.0, t2)
    t = cute.math.fma(t3, 1.0, t12)
    s, c = _sincos_turns(t, (lo1 + lo2) + lo3)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={
        "pos": ColVecLoad("pos"),
        "dpos": ColVecLoad("dpos"),
        "freq": RowVecLoad("freq"),
        "logz": RowVecLoad("logz"),
    },
    mode="acc_pair",
)
def xpos_posfreq_epi(acc, pos, dpos, freq, logz, bias):
    """xPos (length-extrapolatable RoPE, Sun et al. 2022): the rope_posfreq
    rotation times a per-pair magnitude decay zeta_j**dpos.

    xPos REQUIRES OPPOSITE scales on Q and K — query zeta^(+m/s), key
    zeta^(-m/s), so the attention score decays as zeta^((m-n)/s) — and that
    sign lives in the DATA, not in two kernels: ``logz`` carries per-column
    log2(zeta_j) (make_xpos_log_scale), positive over Q columns, negative
    over K, zero over V (exp2(0) = 1, so the packed-QKV passthrough
    survives). ``dpos`` is the decay exponent, typically
    (pos - offset) / scale_base — a separate colvec from the rotation
    position because implementations offset/rescale it to keep zeta**dpos in
    f32 range. Costs one MUFU.EX2 per pair: the quarter-rate XU pipe goes
    from 2 to 3 ops/pair, so expect ~1.5x plain rope's exposed-epilogue
    overhead (hidden regimes unaffected)."""
    x1, x2 = unpack(acc + bias)
    s, c = _sincos_turns(*_angle_turns(pos, freq))
    z, _ = unpack(logz)  # both lanes of a pair share the decay
    zeta = cute.math.exp2(dpos * z, fastmath=True)
    x1, x2 = x1 * zeta, x2 * zeta
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


def _ff_turns_head(inv_freq):
    """(head,) float-float split of inv_freq/2pi, interleaved hi/lo per pair."""
    import math

    import torch

    f64 = inv_freq.double() / (2 * math.pi)
    hi = f64.float()
    lo = (f64 - hi.double()).float()
    return torch.stack([hi, lo], dim=-1).reshape(-1)


def make_interleaved_inv_freq(inv_freq, rotary_n, nonrotary_n=0, head_dim=None):
    """Pack HF-style inv_freq (rotary_dim/2,) into the (rotary_n + nonrotary_n,)
    float-float table rope_posfreq_epi consumes: rotation pair j of each head
    reads (hi, lo) at columns (2j, 2j+1), where hi + lo is the two-float32
    split of inv_freq[j] / 2pi — the kernel's angle unit is TURNS (see
    rope_posfreq_epi) — with heads repeating the pattern over the first
    rotary_n columns. Zero frequency means angle 0, an exact identity (MUFU
    sincos(0) is exactly (0, 1)), which encodes the popular non-rotated
    layouts as data:

    * packed QKV — RoPE on Q/K, V passed through bitwise: the last
      nonrotary_n columns are zero-frequency;
      ``make_interleaved_inv_freq(inv_freq, q_dim + k_dim, v_dim)``.
    * partial rotary (GPT-J / Phi / GLM: rotary_dim < head_dim) — pass
      ``head_dim``: each head's tail beyond 2*len(inv_freq) is zero-frequency.

    NTK-aware scaling (fixed or dynamic) and YaRN's NTK-by-parts ramp are
    pure inv_freq transforms — pass the transformed values here (YaRN's
    mscale additionally needs rope_posfreq_scaled_epi).

    The split preserves inv_freq/2pi to ~2^-49 relative precision — pass
    inv_freq as float64 if available; float32 input still splits exactly (the
    /2pi quotient is computed in f64), only the input's own quantization is
    unrecoverable."""
    import torch

    head = _ff_turns_head(inv_freq)
    rotary_dim = head.shape[0]
    head_dim = rotary_dim if head_dim is None else head_dim
    assert head_dim >= rotary_dim and head_dim % 2 == 0
    head = torch.nn.functional.pad(head, (0, head_dim - rotary_dim))
    assert rotary_n % head_dim == 0, "rotary_n must be a whole number of heads"
    assert nonrotary_n % 2 == 0, "nonrotary_n must be even (columns are processed in pairs)"
    return torch.nn.functional.pad(head.repeat(rotary_n // head_dim), (0, nonrotary_n))


def make_mrope_inv_freq(inv_freq, sections, rotary_n, nonrotary_n=0):
    """Per-axis freq tables for mrope_posfreq_epi (Qwen2-VL mRoPE).

    ``sections`` is the pairs-per-axis split of the head (HF mrope_section,
    e.g. (16, 24, 24) for head_dim 128); axis a's table carries inv_freq/2pi
    over its own section's pairs and zero elsewhere, so the kernel's
    three-term angle dot product selects the right position axis per column
    purely through the data. Returns len(sections) tables shaped like
    make_interleaved_inv_freq's output."""
    import torch

    assert sum(sections) == inv_freq.shape[0], "sections must cover all rotation pairs"
    head = _ff_turns_head(inv_freq)
    head_dim = head.shape[0]
    assert rotary_n % head_dim == 0, "rotary_n must be a whole number of heads"
    assert nonrotary_n % 2 == 0, "nonrotary_n must be even (columns are processed in pairs)"
    outs = []
    start = 0
    for sec in sections:
        masked = torch.zeros_like(head)
        masked[2 * start : 2 * (start + sec)] = head[2 * start : 2 * (start + sec)]
        outs.append(torch.nn.functional.pad(masked.repeat(rotary_n // head_dim), (0, nonrotary_n)))
        start += sec
    return tuple(outs)


def make_xpos_log_scale(head_dim, q_n, k_n=0, nonscaled_n=0, gamma=0.4, device=None):
    """Per-column log2(zeta) table for xpos_posfreq_epi.

    zeta_j = (2j + gamma*head_dim) / ((1 + gamma)*head_dim) per rotation pair
    (the torchscale convention, gamma=0.4). The first q_n columns get
    +log2(zeta) (query side, decay zeta^+dpos), the next k_n get -log2(zeta)
    (key side, zeta^-dpos), the last nonscaled_n get 0 (exp2(0) = 1 — the V
    block of a packed QKV projection passes through unscaled). For separate
    Q / K projections build two tables (q_n=n / k_n=n)."""
    import torch

    assert q_n % head_dim == 0 and k_n % head_dim == 0, "q_n/k_n must be whole heads"
    assert nonscaled_n % 2 == 0, "nonscaled_n must be even (columns are processed in pairs)"
    j2 = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
    zeta = (j2 + gamma * head_dim) / ((1 + gamma) * head_dim)
    logz = torch.log2(zeta).float().repeat_interleave(2)  # both pair lanes share it
    return torch.cat(
        [
            logz.repeat(q_n // head_dim),
            -logz.repeat(k_n // head_dim),
            logz.new_zeros(nonscaled_n),
        ]
    )


def make_interleaved_cos_sin(cos, sin):
    """Interleave HF-style cos/sin (seqlen_ro, head_dim/2) into the
    (seqlen_ro, head_dim) table RotaryCosSinLoad consumes (cos at even
    columns, sin at odd). From the epirope hand-written kernel."""
    import torch

    assert cos.shape == sin.shape and cos.ndim == 2
    return torch.stack([cos, sin], dim=-1).reshape(cos.shape[0], -1).contiguous()
