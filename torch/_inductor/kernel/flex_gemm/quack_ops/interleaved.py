# mypy: allow-untyped-defs
"""Interleaved two-lane epilogue tensors: ``t[m, 2 * n + j]`` is lane ``j`` at ``acc[m, n]``.

``InterleavedTileLoad`` reads such an ``[M, 2N]`` capture and
``InterleavedMainStore`` writes such an ``[M, 2N]`` main output from an
``[M, N]`` accumulator (the inverse of ``GroupedMainStore``'s contraction). Both
move lane-fastest register fragments through TMA tiles doubled along N, so a GLU
backward can read the saved interleaved ``h`` and write ``dh`` in place.
"""

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from torch._inductor.kernel.flex_gemm.quack_ops.main_store import _flatten, _map_strides
from torch._vendor.quack import copy_utils
from torch._vendor.quack.epilogue.ops import _TileLoadState, TileLoad, TileStore


INTERLEAVED_LANES = 2


def _interleave_n_layout(layout):
    """Map an N tile layout ``n -> col`` to the lane pair ``(lane, n) -> 2 col + lane``.

    A dense ``2 * size`` layout is wrong when ``layout`` is a permutation (the
    warp_n=2 epi tile of 2-CTA 64-row CTA tiles and its register-copy tiler).
    """
    return cute.coalesce(
        cute.logical_product(cute.make_layout(INTERLEAVED_LANES), layout)
    )


def _interleaved_epi_tile(gemm, epi_tile):
    """Double an epilogue tile's N extent for the two interleaved lanes."""
    return (epi_tile[0], _interleave_n_layout(epi_tile[1]))


def _interleave_tv(layout_tv, tile_m):
    """Prepend a lane value mode (the next N column) and double N strides.

    The register-to-smem TV layout indexes its tile as ``m + tile_m * n``; each
    accumulator value at column ``n`` becomes the pair at columns ``2n, 2n + 1``.
    """
    thr_shape, val_shape = layout_tv.shape
    thr_stride, val_stride = layout_tv.stride

    def col(st):
        return st if st < tile_m else st * INTERLEAVED_LANES

    vals = _flatten(val_shape, val_stride)
    return cute.make_layout(
        (thr_shape, (INTERLEAVED_LANES, *(sh for sh, _ in vals))),
        stride=(
            _map_strides(thr_shape, thr_stride, col),
            (tile_m, *(col(st) for _, st in vals)),
        ),
    )


def _interleaved_tiled_copy(tiled_copy_tv, tiler_mn, dtype):
    """Universal-atom copy of a lane-fastest fragment over an N-doubled tile."""
    tiler_m, tiler_n = tiler_mn
    layout_tv = _interleave_tv(tiled_copy_tv, cute.size(tiler_m))
    # Each thread's two lanes are adjacent columns, so copy them as one vector.
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=dtype.width * INTERLEAVED_LANES,
    )
    tiler = (tiler_m, _interleave_n_layout(tiler_n))
    return cute.make_tiled_copy(atom, layout_tv, tiler)


@cute.jit
def interleaved_lane(t, lane, ref):
    """Return lane ``lane`` of an interleaved fragment with ``ref``'s shape."""
    size = cute.size(ref.shape)
    return t.reshape((INTERLEAVED_LANES, size))[lane, None].reshape(ref.shape)


@cute.jit
def interleave_lanes(a, b):
    """Pack two same-shape lane fragments lane-fastest for ``InterleavedMainStore``."""
    size = cute.size(a.shape)
    out = cute.make_rmem_tensor(cute.make_layout((INTERLEAVED_LANES, size)), a.dtype)
    a_flat = a.reshape((size,))
    b_flat = b.reshape((size,))
    for i in cutlass.range_constexpr(size):
        out[0, i] = a_flat[i]
        out[1, i] = b_flat[i]
    return out.load()


def _require_n_major(tensor, what):
    # Lane pairs are adjacent N columns; the TV layouts below assume N-major.
    if not cutlass.utils.LayoutEnum.from_tensor(tensor).is_n_major_c():
        raise ValueError(f"{what} must be N-major")


def interleaved_supports_config(config):
    """SM100 only; under swap_ab the lane pairs would be M-adjacent, not N-adjacent."""
    return config.device_capacity == 10 and not config.swap_ab


class InterleavedMainStore(TileStore):
    """Store a lane-interleaved ``[M, 2N]`` main output from an ``[M, N]`` accumulator."""

    supports_swap_ab = False

    def __init__(self, name):
        super().__init__(name, epi_tile_fn=_interleaved_epi_tile)

    def output_n(self, n):
        """Return the expanded logical output N extent."""
        return n * INTERLEAVED_LANES

    def supports_config(self, config):
        return interleaved_supports_config(config)

    def config_support_error(self, configs):
        return "interleaved main outputs require an SM100 config without swap_ab"

    def to_params(self, gemm, args):
        _require_n_major(getattr(args, self.name), "interleaved main output")
        return super().to_params(gemm, args)

    def _make_tiled_copy_r2s(self, gemm, params, tiled_copy_r2s, tiled_copy_t2r):
        return _interleaved_tiled_copy(
            tiled_copy_r2s.layout_src_tv_tiled,
            tiled_copy_r2s.tiler_mn,
            getattr(gemm, self._dtype_gemm_attr()),
        )

    @cute.jit
    def store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
        # The lane-fastest fragment already follows the copy's value order;
        # reshape it to the smem partition instead of retiling a TMEM layout.
        frag = cute.make_rmem_tensor(
            cute.make_layout(tRS_s_stage.shape), frag_out.element_type
        )
        frag.store(frag_out.load().reshape(frag.shape))
        cute.copy(tiled_copy, frag, tRS_s_stage)

    def store_tile_shape_mn(self, gemm):
        return (
            gemm.cta_tile_shape_mnk[0],
            gemm.cta_tile_shape_mnk[1] * INTERLEAVED_LANES,
        )


class InterleavedTileLoad(TileLoad):
    """Load an interleaved ``[M, 2N]`` capture as a lane-fastest register fragment."""

    supports_swap_ab = False

    def __init__(self, name, dtype=None):
        super().__init__(name, epi_tile_fn=_interleaved_epi_tile, dtype=dtype)

    def supports_config(self, config):
        return interleaved_supports_config(config)

    def config_support_error(self, configs):
        return "interleaved captures require an SM100 config without swap_ab"

    def to_params(self, gemm, args):
        _require_n_major(getattr(args, self.name), "interleaved capture")
        return super().to_params(gemm, args)

    def load_g2s_copy_fn(
        self, gemm, params, smem_tensor, tile_coord_mnkl, varlen_manager, epi_pipeline
    ):
        tensor = getattr(params, self.name)
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            getattr(params, self._tma_atom_key()),
            varlen_manager.offset_batch_epi(tensor, tile_coord_mnkl[3]),
            (
                gemm.cta_tile_shape_mnk[0],
                gemm.cta_tile_shape_mnk[1] * INTERLEAVED_LANES,
            ),
            getattr(params, self._epi_tile_key()),
            smem_tensor,
            tile_coord_mnkl,
        )
        return copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        if const_expr(gemm.arch != 100):
            raise NotImplementedError("interleaved captures require the SM100 epilogue")
        tiled_copy_s2r = _interleaved_tiled_copy(
            ctx.tiled_copy_t2r.layout_dst_tv_tiled,
            ctx.tiled_copy_t2r.tiler_mn,
            getattr(gemm, self._dtype_gemm_attr()),
        )
        tSR_sTile = tiled_copy_s2r.get_slice(ctx.tidx).partition_S(smem_tensor)
        tSR_rTile = cute.make_rmem_tensor(
            cute.make_layout(tSR_sTile[None, None, None, 0].shape),
            getattr(gemm, self._dtype_gemm_attr()),
        )
        return _TileLoadState(tiled_copy_s2r, tSR_rTile, tSR_rTile, tSR_sTile)
