# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable, TypeAlias, Tuple
from dataclasses import dataclass
import enum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import min as dsl_min

from ..quack import layout_utils
from . import utils as utils
from .block_info import BlockInfo
from .seqlen_info import SeqlenInfoQK
from .utils import AuxData

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def call_mask_mod(
    mask_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_data: AuxData,
):
    # Compatibility shim for pre-aux_scalars mask_mod callables.
    if const_expr(aux_data.scalars is not None):
        return mask_mod(
            batch_idx,
            head_idx,
            q_idx,
            kv_idx,
            seqlen_info,
            aux_data.tensors,
            aux_data.scalars,
        )
    return mask_mod(
        batch_idx,
        head_idx,
        q_idx,
        kv_idx,
        seqlen_info,
        aux_data.tensors,
    )


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions >= limit (inclusive lower bound).

    Positions limit..31 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
    """Transform SM90 MMA column coordinate to R2P element index.

    SM90 MMA accumulator column indices are non-contiguous: 0, 1, 8, 9, 16, 17, ...
    Element indices are contiguous: 0, 1, 2, 3, 4, 5, ...
    This converts a column-space threshold to element-space for r2p_bitmask_below/above.
    """
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    """Convert a row coordinate to an R2P element index in the warp-group interleaved layout.

    In the SM100 backward pass, 2 warp groups share TMEM. The TMEM load atom
    distributes rows in an interleaved pattern: elements 0..num_rep-1 map to
    rows 0..num_rep-1 (warp group 0), elements num_rep..2*num_rep-1 map to
    rows num_rep*num_wg..num_rep*num_wg+num_rep-1 (warp group 1), and so on.
    Row-coordinate thresholds (causal limits, window bounds, uih_len) must be
    converted to element indices before use with r2p_bitmask_above/below.

    Rows not owned by this thread (in the gap between warp groups) are clamped
    to the boundary element index, which is safe because R2P thresholds are
    monotonic.

    Example with num_rep=16, num_wg=2:
        row  0 -> elem  0,  row 15 -> elem 15,
        row 16 -> elem 16 (clamped), row 31 -> elem 16 (clamped),
        row 32 -> elem 16, row 33 -> elem 17, row 47 -> elem 31.
    """
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@cute.jit
def apply_packed_mask_chunk(
    X: cute.Tensor,
    chunk_idx: cutlass.Constexpr[int],
    mask: Uint32,
) -> None:
    """Apply one 32-bit keep mask to one 32-column chunk.

    The one-iteration chunk loop keeps the same lowering pattern as mask_r2p_lambda.
    """
    ncol = const_expr(cute.size(X.shape))
    col_base = chunk_idx * MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(1):
        for i in cutlass.range_constexpr(
            min(MASK_R2P_CHUNK_SIZE, ncol - col_base - s * MASK_R2P_CHUNK_SIZE)
        ):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = col_base + s * MASK_R2P_CHUNK_SIZE + i
            X[c] = X[c] if in_bound else -Float32.inf


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                r2p = const_expr(not self.swap_AB)
                if const_expr(not r2p):
                    # traverse column index.
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                    mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

        elif const_expr(
            not mask_causal and not mask_local and mask_mod is not None
        ):  # FlexAttention mask mod
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            has_fastdiv = const_expr(
                fastdiv_mods is not None
                and fastdiv_mods[0] is not None
                and fastdiv_mods[1] is not None
            )
            wrap_aux_indices = const_expr(
                has_fastdiv and mask_seqlen and const_expr(aux_data.tensors is not None)
            )

            for r in cutlass.range_constexpr(nrow):
                # Respect swap_AB: ROW/COL determine which coordinate component corresponds to Q/KV.
                local_row = tScS_mn[r, 0][ROW]
                global_row_idx = local_row + m_block * self.tile_m
                row_for_mod = global_row_idx
                head_idx_for_mod = head_idx
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    head_offset = global_row_idx % self.qhead_per_kvhead_packgqa
                    head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    row_for_mod = global_row_idx // self.qhead_per_kvhead_packgqa
                row_for_seqlen = row_for_mod
                if const_expr(wrap_aux_indices):
                    _, row_for_mod = divmod(row_for_mod, fastdiv_mods[0])

                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][COL]
                    # Convert to absolute column index
                    global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
                    col_for_mod = global_col_idx
                    if const_expr(wrap_aux_indices):
                        _, col_for_mod = divmod(global_col_idx, fastdiv_mods[1])

                    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                    head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
                    q_idx_ssa = utils.scalar_to_ssa(row_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(col_for_mod, cutlass.Int32)
                    mask_value = call_mask_mod(
                        mask_mod,
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_data,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    if const_expr(mask_seqlen):
                        out_of_bounds = (row_for_seqlen >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                    else:
                        acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf

        else:  # Causal or local
            if const_expr(not self.swap_AB):
                # If PackGQA, we split the work of compute divmod among threads in the same row
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(not self.swap_AB)  # R2P trick, see apply_mask_sm100
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            col_limit_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            mask_r2p_lambda(
                                acc_S_mn[r, None],
                                lambda s: r2p_bitmask_below(col_limit_r2p, s),
                                rank1=True,
                            )
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    r2p_local = const_expr(not self.swap_AB)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        if const_expr(not r2p_local):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                col_idx = t0ScS_mn[0, c][1]
                                if col_idx >= col_limit_right or col_idx < col_limit_left:
                                    acc_S_mn[r, c] = -Float32.inf
                        else:
                            col_limit_right_r2p = sm90_col_to_r2p_idx(col_limit_right)
                            col_limit_left_r2p = sm90_col_to_r2p_idx(col_limit_left)

                            def mask_gen_fn(s: int) -> Uint32:
                                return r2p_bitmask_below(
                                    col_limit_right_r2p, s
                                ) & r2p_bitmask_above(col_limit_left_r2p, s)

                            mask_r2p_lambda(acc_S_mn[r, None], mask_gen_fn, rank1=True)
            else:  # swap_AB
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else (
                                col0 - causal_row_offset - self.window_size_right
                                if const_expr(self.window_size_right is not None)
                                else 0
                            )
                        )
                        row_limit_bot = (
                            col0 - causal_row_offset + self.window_size_left
                            if const_expr(self.window_size_left is not None)
                            else self.tile_m
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_mod_sm100_scalar(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_mod: cutlass.Constexpr[Callable],
        batch_idx: Int32,
        head_idx: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """Apply a scalar FlexAttention mask_mod to an SM100 accumulator fragment.

        Each accumulator lane calls mask_mod once with logical (batch, head, q, kv)
        indices. Pack-GQA rows are converted back to logical q/head indices before
        the call. When aux tensors are present, indices are wrapped with fastdiv so
        mask_mod never reads outside the per-example auxiliary storage.
        """
        has_fastdiv = const_expr(
            fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None
        )
        batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
        ncol = const_expr(cute.size(tScS_t2r.shape))

        for i in cutlass.range_constexpr(ncol):
            row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
            col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
            global_row = row_coord + m_block * self.tile_m
            global_col = col_coord + n_block * self.tile_n

            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert head_divmod is not None
                mask_row, head_offset = divmod(global_row, head_divmod)
                head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
            else:
                head_idx_for_mod = head_idx
                mask_row = global_row

            mask_row_for_mod = mask_row
            if const_expr(has_fastdiv and aux_data.tensors is not None):
                if check_q_boundary:
                    _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])
            global_col_for_mod = global_col
            if const_expr(has_fastdiv and mask_seqlen and aux_data.tensors is not None):
                _, global_col_for_mod = divmod(global_col, fastdiv_mods[1])

            head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32)
            mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32)
            kv_idx_ssa = utils.scalar_to_ssa(global_col_for_mod, cutlass.Int32)
            mask_value = call_mask_mod(
                mask_mod,
                batch_idx_ssa,
                head_idx_ssa,
                mask_row_ssa,
                kv_idx_ssa,
                self.seqlen_info,
                aux_data,
            )
            cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
            acc_S[i] = acc_S[i] if cond else -Float32.inf
            if const_expr(mask_seqlen):
                acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
            if check_q_boundary:
                acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]

    @cute.jit
    def apply_mask_mod_sm100_vector(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_mod: cutlass.Constexpr[Callable],
        batch_idx: Int32,
        head_idx: Int32,
        vec_size: cutlass.Constexpr[int],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """Apply a vectorized FlexAttention mask_mod to an SM100 fragment.

        mask_mod receives vec_size adjacent KV indices for one logical q row and
        returns bit-packed Uint32 keep masks. Low bits correspond to lower KV
        indices. The packed masks are combined with sequence-boundary checks, then
        applied in 32-column chunks so the final masking lowers to R2P.
        """
        has_fastdiv = const_expr(
            fastdiv_mods is not None and fastdiv_mods[0] is not None and fastdiv_mods[1] is not None
        )
        batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
        ncol = const_expr(cute.size(tScS_t2r.shape))
        mask_vals_per_apply = const_expr(max(1, vec_size // 32))
        calls_per_apply = const_expr(max(1, 32 // vec_size))
        n_calls = const_expr(cute.ceil_div(ncol, vec_size))
        mask_vals = cute.make_rmem_tensor(mask_vals_per_apply, dtype=cutlass.Uint32)

        # Accumulate enough vector mask_mod calls to produce 32-bit chunks that
        # apply_packed_mask_chunk can lower to R2P.
        for s in cutlass.range_constexpr(n_calls):
            if const_expr(s % calls_per_apply == 0):
                for c in cutlass.range_constexpr(mask_vals_per_apply):
                    mask_vals[c] = cutlass.Uint32(0)
            i = s * vec_size
            row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
            col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
            global_row = row_coord + m_block * self.tile_m
            global_col = col_coord + n_block * self.tile_n
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert head_divmod is not None
                mask_row, head_offset = divmod(global_row, head_divmod)
                head_idx_for_mod = head_idx * self.qhead_per_kvhead_packgqa + head_offset
            else:
                head_idx_for_mod = head_idx
                mask_row = global_row
            mask_row_for_mod = mask_row
            if const_expr(has_fastdiv and aux_data.tensors is not None):
                if check_q_boundary:
                    _, mask_row_for_mod = divmod(mask_row, fastdiv_mods[0])

            head_idx_ssa = utils.scalar_to_ssa(head_idx_for_mod, cutlass.Int32).broadcast_to(
                (vec_size,)
            )
            mask_row_ssa = utils.scalar_to_ssa(mask_row_for_mod, cutlass.Int32).broadcast_to(
                (vec_size,)
            )
            batch_idx_ssa_call = batch_idx_ssa.broadcast_to((vec_size,))
            kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

            # Build the per-lane KV indices for this vectorized mask_mod call.
            for j in cutlass.range_constexpr(min(vec_size, ncol - i)):
                col_j_coord = tScS_t2r[i + j][1] if not self.swap_AB else tScS_t2r[i + j][0]
                col_j_global = col_j_coord + n_block * self.tile_n
                col_j_for_mod = col_j_global
                if const_expr(has_fastdiv and mask_seqlen and aux_data.tensors is not None):
                    _, col_j_for_mod = divmod(col_j_global, fastdiv_mods[1])
                kv_idx_vec[j] = col_j_for_mod
            kv_idx_ssa = kv_idx_vec.load()

            # mask_value is already bit-packed by the vectorized mask_mod.
            mask_value = call_mask_mod(
                mask_mod,
                batch_idx_ssa_call,
                head_idx_ssa,
                mask_row_ssa,
                kv_idx_ssa,
                self.seqlen_info,
                aux_data,
            )

            # For vec_size < 32, multiple mask_mod calls fill one R2P chunk.
            bit_offset = const_expr((s % calls_per_apply) * vec_size)
            seqlen_thresh_call = (
                self.seqlen_k - global_col if const_expr(mask_seqlen) else cutlass.Int32(0)
            )
            q_in_bounds = mask_row < self.seqlen_q if check_q_boundary else cutlass.Boolean(True)
            for c in cutlass.range_constexpr(mask_vals_per_apply):
                mask_val = mask_value[c]
                if const_expr(vec_size < 32):
                    lane_keep = utils.shr_u32(
                        cutlass.Uint32(0xFFFFFFFF),
                        cutlass.Uint32(32 - vec_size),
                    )
                    mask_val = mask_val & lane_keep
                if const_expr(mask_seqlen):
                    mask_val = mask_val & r2p_bitmask_below(seqlen_thresh_call, c)
                if check_q_boundary:
                    mask_val = mask_val if q_in_bounds else cutlass.Uint32(0)
                mask_vals[c] = mask_vals[c] | (mask_val << bit_offset)

            # Apply only when the 32-bit chunk is complete, or at the tile tail.
            is_last_in_apply = const_expr(s % calls_per_apply == calls_per_apply - 1)
            is_last_overall = const_expr(s == n_calls - 1)
            if const_expr(is_last_in_apply or is_last_overall):
                apply_idx = s // calls_per_apply
                for c in cutlass.range_constexpr(mask_vals_per_apply):
                    chunk_idx = apply_idx * mask_vals_per_apply + c
                    # Skip packed chunks that start past the accumulator fragment.
                    if const_expr(chunk_idx * 32 < ncol):
                        apply_packed_mask_chunk(acc_S, chunk_idx, mask_vals[c])

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        vec_size: cutlass.Constexpr[int] = 1,
        check_q_boundary: bool = False,
        r2p: bool = True,
        rBitmask: Optional[cute.Tensor] = None,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n

        if const_expr(rBitmask is not None):
            ncol_packed = const_expr(cute.size(rBitmask.shape[0]))
            for i in cutlass.range_constexpr(ncol_packed):
                col_start = 32 * i  # mask is bit-packed into uint32
                curr_mask_val = rBitmask[i]
                for j in cutlass.range_constexpr(32):
                    curr_col = col_start + j
                    mask = (curr_mask_val >> j) & 1
                    acc_S[curr_col] = acc_S[curr_col] if cutlass.Boolean(mask) else -Float32.inf

        elif const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
                        rank1=True,
                    )

        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # FlexAttention mask_mod vectorization is gated on `mask_mod.__vec_size__`.
            # vec_size == 1 returns a scalar Boolean. vec_size > 1 returns packed
            # Uint32 mask fragments: one word per 32 evaluated columns.
            assert vec_size % 32 == 0 or 32 % vec_size == 0, (
                "vec_size must divide 32 or be a multiple of 32"
            )
            if const_expr(vec_size == 1):
                self.apply_mask_mod_sm100_scalar(
                    acc_S,
                    tScS_t2r,
                    m_block,
                    n_block,
                    mask_seqlen,
                    mask_mod,
                    batch_idx,
                    head_idx,
                    aux_data,
                    fastdiv_mods,
                    head_divmod,
                    check_q_boundary,
                )
            else:
                self.apply_mask_mod_sm100_vector(
                    acc_S,
                    tScS_t2r,
                    m_block,
                    n_block,
                    mask_seqlen,
                    mask_mod,
                    batch_idx,
                    head_idx,
                    vec_size,
                    aux_data,
                    fastdiv_mods,
                    head_divmod,
                    check_q_boundary,
                )

        else:  # Causal or local
            causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset + 1
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(col_limit_right, s),
                        rank1=True,
                    )
            else:
                local_row_offset_right = (
                    causal_row_offset + 1 + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                if const_expr(not r2p):
                    # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # Dual-bound R2P masking for SM100.
                    # Masks elements where: NOT (col_limit_left <= col < col_limit_right)

                    def mask_gen_fn(s: int) -> Uint32:
                        return r2p_bitmask_below(col_limit_right, s) & r2p_bitmask_above(
                            col_limit_left, s
                        )

                    mask_r2p_lambda(acc_S, mask_gen_fn, rank1=True)

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        """
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.

        Coordinate convention:
        - ROW corresponds to Q (m_block)
        - COL corresponds to KV (n_block)

        is_full_block: If True, skip mask_mod (all elements valid). Only apply seqlen masking.
        check_m_boundary: If False, skip seqlen_q boundary check (optimization for non-boundary m_blocks).
                          When iterating m_blocks in forward order, only the last m_block may be partial.
        """
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        # assert t0ScS_t2r[0][COL] == 0, "col0 == 0" # tmp comment for 2-cta bwd
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case with mask_mod (backward)
            #
            # Coordinate convention: ROW → Q (m_block), COL → KV (n_block).
            # These already account for swap_AB.
            #
            # FULL blocks: mask_mod returns True for all elements, so skip it.
            #   Still need seqlen bounds check (elements may be OOB on last m_block).
            # PARTIAL blocks: apply mask_mod element-wise, then seqlen bounds.
            if is_full_block:
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        # Entire tile is OOB for K
                        for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                            acc_S[i] = -cutlass.Float32.inf
                    elif check_m_boundary:
                        # Last m_block: check Q and K boundaries
                        ncol = const_expr(cute.size(tScS_t2r.shape))
                        for i in cutlass.range_constexpr(ncol):
                            row_coord = tScS_t2r[i][ROW]
                            col_coord = tScS_t2r[i][COL]
                            global_q = row_coord + m_block * self.tile_m
                            global_kv = col_coord + n_block * self.tile_n
                            q_out_of_bounds = global_q >= self.seqlen_q
                            kv_out_of_bounds = global_kv >= self.seqlen_k
                            out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                            acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]
            else:
                # Partial block
                has_fastdiv = const_expr(
                    fastdiv_mods is not None
                    and fastdiv_mods[0] is not None
                    and fastdiv_mods[1] is not None
                )
                wrap_aux_indices = const_expr(
                    has_fastdiv and mask_seqlen and const_expr(aux_data.tensors is not None)
                )
                batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32)
                head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32)

                ncol = const_expr(cute.size(tScS_t2r.shape))
                for i in cutlass.range_constexpr(ncol):
                    row_coord = tScS_t2r[i][ROW]
                    col_coord = tScS_t2r[i][COL]
                    global_q = row_coord + m_block * self.tile_m
                    global_kv = col_coord + n_block * self.tile_n

                    q_idx_for_mod = global_q
                    kv_idx_for_mod = global_kv
                    if const_expr(wrap_aux_indices):
                        _, q_idx_for_mod = divmod(global_q, fastdiv_mods[0])
                        _, kv_idx_for_mod = divmod(global_kv, fastdiv_mods[1])

                    q_idx_ssa = utils.scalar_to_ssa(q_idx_for_mod, cutlass.Int32)
                    kv_idx_ssa = utils.scalar_to_ssa(kv_idx_for_mod, cutlass.Int32)

                    mask_value = call_mask_mod(
                        mask_mod,
                        batch_idx_ssa,
                        head_idx_ssa,
                        q_idx_ssa,
                        kv_idx_ssa,
                        self.seqlen_info,
                        aux_data,
                    )
                    cond = cutlass.Boolean(utils.ssa_to_scalar(mask_value))
                    acc_S[i] = acc_S[i] if cond else -cutlass.Float32.inf

                    if const_expr(mask_seqlen):
                        # check_m_boundary=False skips q check for non-boundary m_blocks
                        q_out_of_bounds = check_m_boundary and (global_q >= self.seqlen_q)
                        kv_out_of_bounds = global_kv >= self.seqlen_k
                        out_of_bounds = q_out_of_bounds or kv_out_of_bounds
                        acc_S[i] = -cutlass.Float32.inf if out_of_bounds else acc_S[i]

        elif const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:  # Causal or local
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                # tidx = cute.arch.thread_idx()[0] % 256
                # if tidx < 32:
                #     cute.printf("tidx = {}, {} {}, {} {}", tidx, tScS_t2r[0][0], tScS_t2r[0][1], tScS_t2r[1][0], tScS_t2r[1][1])
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    # If col is beyond the column limit, we want to mask out the entire
                    # column, by setting row limit to be self.tile_m.
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])  # 16 or 32
                    num_wg = 2
                    row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_above(row_limit, s),
                        rank1=True,
                    )
            else:
                if const_expr(self.window_size_right is not None):
                    row_limit_top = causal_offset - self.window_size_right
                else:
                    row_limit_top = 0
                if const_expr(self.window_size_left is not None):
                    row_limit_bot = causal_offset + self.window_size_left
                if const_expr(mask_seqlen):
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m
                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        row_idx = t0ScS_t2r[i][ROW]
                        local_mask = row_idx < row_limit_top
                        if const_expr(self.window_size_left is not None):
                            local_mask |= row_idx > row_limit_bot
                        acc_S[i] = -cutlass.Float32.inf if local_mask else acc_S[i]
                else:

                    def mask_gen_fn(s: int) -> Uint32:
                        num_rep = cute.size(tScS_t2r, mode=[0])
                        num_wg = 2

                        row_limit = row_to_r2p_idx(row_limit_top, num_rep, num_wg)
                        mask = r2p_bitmask_above(row_limit, s)

                        if const_expr(self.window_size_left is not None):
                            row_limit_bottom = row_to_r2p_idx(row_limit_bot + 1, num_rep, num_wg)
                            mask = mask & r2p_bitmask_below(row_limit_bottom, s)

                        return mask

                    mask_r2p_lambda(
                        acc_S,
                        mask_gen_fn,
                        rank1=True,
                    )


# -----------------------------------------------------------------------------
# SM100 FMHA fused-mask policy layer (separate from generic mask primitives).
# -----------------------------------------------------------------------------


class Sm100MaskEnum(enum.Enum):
    """Enumeration of mask types for FMHA operations.

    - RESIDUAL_MASK: Residual mask for handling variable sequence lengths
    - WINDOW_MASK: Window mask for attention which also includes causal and no mask
    - WINDOW_MASK_INFERENCE: Same as the window mask, but has the limitation that the end of q is aligned with the end of k
    - WINDOW_MASK_BWD: Window mask for backward pass
    - WINDOW_MASK_BWD_INFERENCE: Same as the window mask for backward pass, but has the limitation that the end of q is aligned with the end of k
    """

    NO_MASK = enum.auto()
    RESIDUAL_MASK = enum.auto()
    CAUSAL_MASK = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    # Deprecated the following types
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()


class Sm100FusedMask:
    """A fused mask implementation for FMHA operations.

    This class handles different types of attention masks including no mask,
    residual mask for variable sequence lengths, and causal mask for
    autoregressive attention patterns.

    The class provides methods to:
    - Calculate trip counts for different mask types
    - Apply masks to attention scores
    - Handle masked and unmasked trip calculations
    """

    def get_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of trips needed for the current block.

        The trip count depends on the mask type and the block coordinates.
        For causal masks, it considers the autoregressive constraint.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of trips needed.
        :rtype: Int32
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == Sm100MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is Sm100MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK
            or mask_type == Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = dsl_min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(
            mask_type == Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type == Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = dsl_min(max_blocks_q, tmp_blocks_q)
        start_block = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start_count_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        return n_block_min, n_block_max - n_block_min

    @cute.jit
    def get_trip_mask_bounds_via_block_info(
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """Return SM100-style mask boundaries for dense iteration.

        Returns:
          - n_block_min_causal_local_mask: right-side masked region start
          - n_block_min_before_local_mask: start of fully unmasked middle region
        """
        block_info = BlockInfo(
            tile_m=tile_shape[0],
            tile_n=tile_shape[1],
            is_causal=is_causal,
            is_local=is_local and not is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        seqlen_info = SeqlenInfoQK(
            offset_q=Int32(0),
            offset_k=Int32(0),
            padded_offset_q=Int32(0),
            padded_offset_k=Int32(0),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_offset=Int32(0),
            block_idx_offset=Int32(0),
            num_n_blocks=cute.ceil_div(seqlen_k, tile_shape[1]),
            has_cu_seqlens_q=False,
            has_cu_seqlens_k=False,
            has_seqused_q=False,
            has_seqused_k=False,
        )
        n_block_min, _ = block_info.get_n_block_min_max(seqlen_info, blk_coord[0])
        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen_info, blk_coord[0], n_block_min
        )
        return n_block_min_causal_local_mask, n_block_min_before_local_mask

    @cute.jit
    def get_trip_start(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Get the start of the trip for the current block.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """
        Get the begin and end tile idx for the leading mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the leading mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                leading_mask_end = dsl_min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_BWD
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Optional[Int32], Optional[Int32]]:
        """
        Get the begin and end tile idx for the trailing mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the trailing mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = Sm100FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = Sm100FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK
            or mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = dsl_min(
                    min_idx_q // tile_shape[1], trip_count + trip_start - 1
                )
                trailing_mask_end = trip_count + trip_start - 1
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                trailing_mask_begin = dsl_min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = dsl_min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of masked trips for the leading mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0
        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Int32:
        """
        Calculate the number of masked trips for the trailing mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param rem_count: Remaining count from previous calculations.
        :type rem_count: Int32

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0

        if cutlass.const_expr(
            mask_type is not Sm100MaskEnum.RESIDUAL_MASK
            and mask_type is not Sm100MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                trailing_mask_begin, trailing_mask_end = Sm100FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = Sm100FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(
                    trailing_mask_begin is not None and trailing_mask_end is not None
                ):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: Sm100MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of unmasked trips for the current block.

        This represents the number of trips that don't require special
        masking treatment.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of unmasked trips.
        :rtype: Int32
        """
        result = (
            Sm100FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - Sm100FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def apply_mask(
        mask_type: Sm100MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """
        Apply the appropriate mask to the attention scores.

        This method modifies the attention scores (acc_qk) based on the mask type
        and the positions in the index tensor.

        :param mask_type: Type of mask to use
        :type mask_type: utils.Sm100MaskEnum
        :param acc_qk: Accumulated QK attention scores tensor.
        :type acc_qk: cute.Tensor
        :param index_qk: Index tensor containing position information.
        :type index_qk: cute.Tensor
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Optional[int]
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[int]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[int]
        """
        offset = 0
        # NOTE: causal masking in this repo aligns the *end* of Q with the *end* of K
        # when seqlen_k != seqlen_q (same as the test/reference implementation):
        #   k_index <= q_index + (seqlen_k - seqlen_q) + window_right
        # In our kernels, causal is represented by (window_left is None, window_right is not None).
        if cutlass.const_expr(window_size_left is None and window_size_right is not None):
            offset = seqlen_k - seqlen_q
        elif cutlass.const_expr(
            mask_type is Sm100MaskEnum.WINDOW_MASK_INFERENCE
            or mask_type is Sm100MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(
                mask_type == Sm100MaskEnum.RESIDUAL_MASK
                or mask_type == Sm100MaskEnum.RESIDUAL_MASK_BWD
            ):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf

    @cute.jit
    def apply_mask_via_causal_local(
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        apply_semantic_window: cutlass.Constexpr[bool] = True,
        is_causal: cutlass.Constexpr[bool] = False,
        is_local: cutlass.Constexpr[bool] = False,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """Apply forward mask without mask_type.

        - If apply_semantic_window=True, apply causal/local window constraints.
        - Always apply residual OOB masking (index_k>=seqlen_k or index_q>=seqlen_q).
        """
        offset = 0
        if cutlass.const_expr(apply_semantic_window):
            # Match WINDOW_MASK_INFERENCE semantics: end-align Q/K when lengths differ.
            offset = seqlen_k - seqlen_q
        for i in cutlass.range_constexpr(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(apply_semantic_window):
                if cutlass.const_expr(is_causal and not is_local):
                    # Pure causal; tolerate both external forms:
                    # - (None, None) from interface
                    # - (None, 0) from fused-mask-style callers
                    right = 0 if const_expr(window_size_right is None) else window_size_right
                    if index_q + offset + right < index_k:
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(
                    is_local or window_size_left is not None or window_size_right is not None
                ):
                    if cutlass.const_expr(window_size_left is None):
                        if index_q + offset + window_size_right < index_k:
                            acc_qk[i] = -Float32.inf
                    elif cutlass.const_expr(window_size_right is None):
                        if index_q + offset - window_size_left > index_k:
                            acc_qk[i] = -Float32.inf
                    else:
                        max_K_index = dsl_min(index_q + offset + window_size_right, seqlen_k)
                        min_K_index = max(0, index_q + offset - window_size_left)
                        if index_k > max_K_index or index_k < min_K_index:
                            acc_qk[i] = -Float32.inf
            # Residual mask is always needed for boundary protection.
            if index_k >= seqlen_k or index_q >= seqlen_q:
                acc_qk[i] = -Float32.inf
