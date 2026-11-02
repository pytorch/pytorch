# mypy: allow-untyped-defs
"""Alignment-safe column captures for variable-length grouped GEMMs."""

import cutlass
import cutlass.cute as cute

import torch
from torch._vendor.quack import utils
from torch._vendor.quack.epilogue.ops import ColVecLoad, is_floating_dtype, Scalar


class ScalarColVecLoad(ColVecLoad):
    """Load sub-32-bit varlen columns without cp.async's 4-byte alignment.

    Selected from static dtype/varlen metadata by the EpiMod factory. The
    inherited needs_async_fence keeps QuACK's epilogue barrier, ordering these
    synchronous shared stores before the inherited begin_loop reads them.
    """

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        dtype = param.element_type
        vector = self._get_gmem_vec(param, ctx)
        tile_m = ctx.tile_M
        tile_idx = ctx.tile_coord_mnkl[0]
        limit = self._valid_extent(vector, tile_idx, tile_m, ctx)
        for i in cutlass.range_constexpr(cute.ceil_div(tile_m, ctx.num_epi_threads)):
            row = ctx.tidx % ctx.num_epi_threads + i * ctx.num_epi_threads
            if row < tile_m:
                value = dtype(0)
                if row < limit:
                    value = vector[tile_idx * tile_m + row]
                # Match cp.async zero-fill for invalid lanes, including M folds.
                smem_tensor[row] = value

        # Keep ColVecLoad's fragment layout/state so begin_loop remains shared.
        shared = ctx.partition_for_epilogue_fn(
            cute.make_tensor(
                smem_tensor.iterator,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0)),
            )
        )
        if cutlass.const_expr(ctx.tiled_copy_t2r is not None):
            shared = ctx.tiled_copy_r2s.retile(shared)
        subtile = cute.group_modes(shared, 3, cute.rank(shared))[None, None, None, 0]
        register_dtype = gemm.acc_dtype if is_floating_dtype(dtype) else dtype
        registers = cute.make_rmem_tensor(subtile.layout, register_dtype)
        return [shared, registers]


class GroupScalar(Scalar):
    """Load a caller-owned TensorWise scale for the current expert."""

    def _validate_pointer_value(self, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"group scalar '{self.name}' must be a tensor")
        if (
            value.ndim != 1
            or value.dtype is not torch.float32
            or not value.is_cuda
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"group scalar '{self.name}' must be contiguous 1-D CUDA Float32"
            )

    def host_validate(self, value, *, batch, **_):
        self._validate_pointer_value(value)
        if value.numel() != batch:
            raise ValueError(
                f"group scalar '{self.name}' must contain one value per group"
            )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        return utils.load_scalar_or_pointer(param + ctx.batch_idx, dtype=self.dtype)
