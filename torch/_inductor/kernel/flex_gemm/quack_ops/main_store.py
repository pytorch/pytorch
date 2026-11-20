# mypy: allow-untyped-defs
"""Grouped-main output store: one value per adjacent-N accumulator group."""

import cutlass
import cutlass.cute as cute

from torch._inductor.kernel.flex_gemm.quack_ops.grouped_reduce import (
    grouped_reduce_supports_config,
)
from torch._vendor.quack.epilogue.ops import setup_epi_tensor, TileStore


def _contract_epi_tile_n(epi_tile, group):
    """Contract an epilogue tile's N extent by a static group size."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(group, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // group)


def _flatten(shape, stride):
    if isinstance(shape, tuple):
        out = []
        for sh, st in zip(shape, stride):
            out += _flatten(sh, st)
        return out
    return [(shape, stride)]


def _map_strides(shape, stride, fn):
    if isinstance(shape, tuple):
        return tuple(_map_strides(sh, st, fn) for sh, st in zip(shape, stride))
    return fn(stride)


def _contract_tv_16dp(layout_tv, tile_m, group):
    """Contract a 16-datapath (M64 1-CTA) t2r register TV layout by ``group`` N lanes.

    Linear tile index is ``m + tile_m * n``. The fragment's fastest value mode
    holds ``group`` adjacent N columns (stride ``tile_m``); the callback folds
    it, so drop that mode and divide every column stride by ``group``.
    """
    thr_shape, val_shape = layout_tv.shape
    thr_stride, val_stride = layout_tv.stride
    vals = _flatten(val_shape, val_stride)
    if vals[0] != (group, tile_m):
        raise NotImplementedError(
            f"grouped main output cannot contract register layout {layout_tv}"
        )

    def col(st):
        if st < tile_m:
            return st
        if st % (group * tile_m):
            raise NotImplementedError(
                f"grouped main output cannot contract register layout {layout_tv}"
            )
        return st // group

    rest = vals[1:] or [(1, 0)]
    return cute.make_layout(
        (thr_shape, tuple(sh for sh, _ in rest)),
        stride=(
            _map_strides(thr_shape, thr_stride, col),
            tuple(col(st) for _, st in rest),
        ),
    )


def _grouped_main_epi_tile_2(gemm, epi_tile):
    """Contract a grouped-main output tile by two adjacent N lanes."""
    return _contract_epi_tile_n(epi_tile, 2)


def _grouped_main_epi_tile_4(gemm, epi_tile):
    """Contract a grouped-main output tile by four adjacent N lanes."""
    return _contract_epi_tile_n(epi_tile, 4)


class GroupedMainStore(TileStore):
    """Store one value per adjacent-N group from a direct TensorSSA callback.

    The callback owns the logical lane contraction; this op owns the contracted
    epilogue tile, output buffer schema, physical store geometry, and QuACK
    config legality. It intentionally remains an ordinary output EpiOp rather
    than introducing a grouped-main GEMM mode.
    """

    supports_swap_ab = False

    def __init__(self, name, group, min_fragment_n=None):
        if group not in (2, 4):
            raise ValueError("GroupedMainStore supports group 2 or 4")
        if min_fragment_n is not None and (
            min_fragment_n <= 0 or min_fragment_n % group
        ):
            raise ValueError("min_fragment_n must be a positive multiple of group")
        epi_tile_fn = (
            _grouped_main_epi_tile_2 if group == 2 else _grouped_main_epi_tile_4
        )
        super().__init__(name, epi_tile_fn=epi_tile_fn)
        self.group = group
        self.min_fragment_n = min_fragment_n

    def config_key(self):
        return (self.group, self.min_fragment_n, *super().config_key())

    def output_n(self, n):
        """Return the contracted logical output N extent."""
        if n % self.group:
            raise ValueError(
                f"grouped main output requires GEMM N divisible by {self.group}, got {n}"
            )
        return n // self.group

    def supports_config(self, config):
        """Return whether a config has validated grouped-main store ownership."""
        # Feed-main fragment reductions need this even without a partial-output sink.
        if self.min_fragment_n is not None and not grouped_reduce_supports_config(
            config, 1, self.min_fragment_n
        ):
            return False
        supported_arch = (
            config.device_capacity in (10, 11)
            if self.group == 2
            else config.device_capacity == 10
        )
        # 1-CTA M64 tiles read TMEM through 16-datapath atoms whose fragments
        # hold column pairs, so only group-2 contraction is expressible there
        # (see _make_tiled_copy_r2s).
        one_cta_m = config.tile_m in (128, 256) or (
            config.tile_m == 64 and self.group == 2
        )
        supported_m_cluster = (one_cta_m and config.cluster_m == 1) or (
            config.tile_m == 256 and config.cluster_m == 2
        )
        min_tile_n = 32 if self.group == 2 else 64
        return (
            supported_arch
            and not config.swap_ab
            and supported_m_cluster
            and config.cluster_n in ((1, 2, 4) if config.tile_m == 64 else (1,))
            and config.tile_n >= min_tile_n
            and config.tile_n % self.group == 0
        )

    def supports_problem(self, config, m, n):
        """Apply problem-size legality not expressible from config fields alone."""
        return (
            n % self.group == 0
            and config.tile_n <= n
            and (self.min_fragment_n is None or n % self.min_fragment_n == 0)
        )

    def config_support_error(self, configs):
        if self.group == 4:
            return "group-4 grouped main outputs require an SM100 config"
        return "group-2 grouped main outputs require an SM100 or SM110 config"

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
        if not layout.is_n_major_c():
            raise ValueError("grouped main output must be N-major")
        setattr(gemm, self._layout_gemm_attr(), layout)
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        epi_tile = _contract_epi_tile_n(gemm.epi_tile, self.group)
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
            self._dtype_field(): tensor.element_type,
        }

    def _make_tiled_copy_r2s(self, gemm, params, tiled_copy_r2s, tiled_copy_t2r):
        tiler_m, tiler_n = tiled_copy_r2s.tiler_mn
        tile_m = cute.size(tiler_m)
        if gemm.cta_tile_shape_mnk[0] != 64 or gemm.use_2cta_instrs:
            return super()._make_tiled_copy_r2s(
                gemm, params, tiled_copy_r2s, tiled_copy_t2r
            )
        # 1-CTA M64 reads TMEM through 16-datapath atoms, where each thread's
        # fragment interleaves two rows. The inherited copy keeps the
        # uncontracted thread-value map, so build the contracted one directly
        # and store through a universal SIMT atom.
        layout_tv = _contract_tv_16dp(
            tiled_copy_r2s.layout_src_tv_tiled, tile_m, self.group
        )
        dtype = getattr(gemm, self._dtype_gemm_attr())
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=dtype.width
        )
        tiler = (tiler_m, cute.make_layout(cute.size(tiler_n) // self.group))
        return cute.make_tiled_copy(atom, layout_tv, tiler)

    def min_epi_tile_n(self, arg_tensor):
        """Keep stores vectorizable and fragment reductions complete."""
        store_width = 0
        if arg_tensor is not None:
            width = arg_tensor.element_type.width
            store_width = self.group * ((128 + width - 1) // width)
        required_n = max(store_width, self.min_fragment_n or 0)
        return required_n or None

    def store_tile_shape_mn(self, gemm):
        return (gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1] // self.group)
