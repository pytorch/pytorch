# mypy: allow-untyped-defs
"""Grouped-main output store: one value per adjacent-N accumulator group."""

import cutlass
import cutlass.cute as cute

from torch._vendor.quack.epilogue.ops import setup_epi_tensor, TileStore


def _contract_epi_tile_n(epi_tile, group):
    """Contract an epilogue tile's N extent by a static group size."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(group, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // group)


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

    def __init__(self, name, group):
        if group not in (2, 4):
            raise ValueError("GroupedMainStore supports group 2 or 4")
        epi_tile_fn = (
            _grouped_main_epi_tile_2 if group == 2 else _grouped_main_epi_tile_4
        )
        super().__init__(name, epi_tile_fn=epi_tile_fn)
        self.group = group

    def config_key(self):
        return (self.group, *super().config_key())

    def output_n(self, n):
        """Return the contracted logical output N extent."""
        if n % self.group:
            raise ValueError(
                f"grouped main output requires GEMM N divisible by {self.group}, got {n}"
            )
        return n // self.group

    def supports_config(self, config):
        """Return whether a config has validated grouped-main store ownership."""
        supported_arch = (
            config.device_capacity in (10, 11)
            if self.group == 2
            else config.device_capacity == 10
        )
        supported_m_cluster = (
            config.tile_m in (128, 256) and config.cluster_m == 1
        ) or (config.tile_m == 256 and config.cluster_m == 2)
        min_tile_n = 64 if self.group == 2 else 128
        return (
            supported_arch
            and not config.swap_ab
            and supported_m_cluster
            and config.cluster_n == 1
            and config.tile_n >= min_tile_n
            and config.tile_n % self.group == 0
        )

    def supports_problem(self, config, m, n):
        """Apply problem-size legality not expressible from config fields alone."""
        return n % self.group == 0 and config.tile_n <= n

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

    def store_tile_shape_mn(self, gemm):
        return (gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1] // self.group)
