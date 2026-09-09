# mypy: allow-untyped-defs
"""Exact typed row gathers: ``ColVecSelect`` with tensor-gather semantics."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass._mlir.dialects import llvm

import torch
from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map
from torch._vendor.quack.epilogue.ops import assume_stride_divisibility, ColVecSelect


_CUTE_TO_TORCH_DTYPE = {
    cute_dtype: torch_dtype for torch_dtype, cute_dtype in torch2cute_dtype_map.items()
}


class _ExactColVecSelectParams(NamedTuple):
    tensor: object
    logical_n: object


class ExactColVecSelect(ColVecSelect):
    """Per-row column selection with exact ``torch.gather`` semantics.

    Upstream ``ColVecSelect`` keeps the cross-entropy contract: an f32 output
    and out-of-range indices that leave prefilled rows untouched. This op
    preserves every selected value, including ``-inf``, traps on indices
    outside ``[0, N)``, and stores directly in ``output_dtype``.
    """

    def __init__(self, name, idx_op, *, output_dtype):
        super().__init__(name, idx_op)
        self.output_dtype = output_dtype
        if output_dtype not in _CUTE_TO_TORCH_DTYPE:
            raise ValueError(
                f"ColVecSelect {name!r}: unsupported output dtype {output_dtype}"
            )
        self._sink_dtype = _CUTE_TO_TORCH_DTYPE[output_dtype]

    def config_key(self):
        return (self.idx_op.cache_key(), self.output_dtype)

    def to_params(self, gemm, args):
        tensor = assume_stride_divisibility(getattr(args, self.name))
        return {self.name: _ExactColVecSelectParams(tensor, gemm.caller_n)}

    def sink_alloc_dtype(self):
        return self._sink_dtype

    def host_validate(
        self, value, *, m, n, tile_M, tile_N, batch, varlen_m, epi_args, **_
    ):
        idx = epi_args.get(self.idx_op.name)
        if idx is None:
            raise ValueError(
                f"sink '{self.name}' requires the '{self.idx_op.name}' index operand"
            )
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"'{self.idx_op.name}' must be int32 or int64, got {idx.dtype}"
            )
        expected = (m,) if varlen_m or batch is None else (batch, m)
        index_shapes = (expected,) if len(expected) == 1 else ((m,), expected)
        if tuple(idx.shape) not in index_shapes:
            raise ValueError(
                f"'{self.idx_op.name}' must have shape in {index_shapes}, got {tuple(idx.shape)}"
            )
        if idx.stride(-1) != 1:
            raise ValueError(f"'{self.idx_op.name}' must have unit innermost stride")
        if tuple(value.shape) != expected:
            raise ValueError(
                f"sink '{self.name}': expected shape {expected}, got {tuple(value.shape)}"
            )
        if value.stride(-1) != 1:
            raise ValueError(f"sink '{self.name}' must have unit innermost stride")
        if torch2cute_dtype_map.get(value.dtype) != self.output_dtype:
            raise ValueError(
                f"sink '{self.name}' must have dtype {self.output_dtype}, got {value.dtype}"
            )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        return super().begin(gemm, param.tensor, smem_tensor, ctx)

    @cute.jit
    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        """Stage strict-gather bounds validation for the finish phase."""
        if const_expr(epi_coord[1] != 0):
            return None
        return (False, (state, epi_coord))

    @cute.jit
    def end_loop_finish(self, gemm, param, staged, tile_coord_mnkl, varlen_manager):
        """Trap strict-gather indices outside the logical output extent."""
        state, epi_coord = staged
        sIdx, _, coords, _, limit_m, n_off, _, _, _ = state
        logical_n = param.logical_n
        coordinates = cute.filter_zeros(
            coords[None, None, None, epi_coord[0], epi_coord[1]]
        )
        for i in cutlass.range(cute.size(coordinates), unroll_full=True):
            row, column = coordinates[i][0], coordinates[i][1]
            index = sIdx[row]
            if (
                row < limit_m
                and n_off == 0
                and column == 0
                and (index < 0 or index >= logical_n)
            ):
                llvm.inline_asm(
                    None,
                    [],
                    "trap;",
                    "",
                    has_side_effects=True,
                    is_align_stack=False,
                )

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        """Store each selected value with direct predicated stores."""
        sIdx, _, coords, gVec, limit_m, n_off, _, _, _, _ = state
        values = cute.filter_zeros(frag)
        coordinates = cute.filter_zeros(coords)
        for i in cutlass.range(cute.size(values), unroll_full=True):
            row, column = coordinates[i][0], coordinates[i][1]
            index = sIdx[row]
            if row < limit_m and n_off + column == index:
                gVec[row] = values[i].to(gVec.element_type)
