# Copyright (c) 2025, Wentao Guo, Tri Dao.
from __future__ import annotations
import math
from typing import NamedTuple, Tuple, Optional, Callable, Type

from torch import Tensor
from torch._subclasses.fake_tensor import is_fake_tensor

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Float32, const_expr
from cutlass.cute.runtime import make_ptr
from cutlass.cute.nvgpu import warp

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import (
    mlir_namedtuple,
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from .epi_composable import ComposableEpiMixin
from .epi_ops import (
    ColVecLoad,
    RowVecLoad,
    RowVecTupleLoad,
    ColVecTupleLoad,
    ScalarTupleLoad,
    EpiSmemBytes,
    GroupedLocalReduce,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    Scalar,
    grouped_local_reduce_uses_smem,
    grouped_rowvec_reduce_value,
    TileTupleLoad,
    TileStore,
)
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .gemm_default_epi import GemmDefaultEpiMixin
from .gemm_tvm_ffi_utils import (
    get_major,
    perm3d_single,
    make_scheduler_args,
    make_varlen_args,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    div_for_dtype,
    make_fake_gemm_tensors,
    compile_gemm_kernel,
)
from .cache import jit_cache
from . import layout_utils
from . import copy_utils
from .layout_utils import permute_gated_Cregs_b16
from .activation import act_fn_map, gate_fn_map
from .rounding import RoundingMode, convert_f32_to_bf16_sr, epilogue_aux_out_sr_seed


_tensor_epilogue_fns: dict[str, Callable] = {}
_local_reduce_combine_fns: dict[str, Callable] = {}
_local_reduce_finalize_fns: dict[str, Callable] = {}


def power_of_2_divisibility(value: int, max_divisibility: int) -> int:
    value = abs(int(value))
    if value == 0:
        return max_divisibility
    divisibility = 1
    while divisibility * 2 <= max_divisibility and value % (divisibility * 2) == 0:
        divisibility *= 2
    return divisibility


def tensor_stride_divisibility(
    tensor: Tensor | None,
    dtype: Type[cutlass.Numeric],
    leading_dim: int,
) -> int:
    """Return the strongest 16-byte-capped layout contract true for a tensor."""
    if tensor is None:
        return 1
    max_divisibility = div_for_dtype(dtype)
    divisibility = max_divisibility
    for dim, stride in enumerate(tensor.stride()):
        if dim != leading_dim:
            divisibility = min(
                divisibility, power_of_2_divisibility(stride, max_divisibility)
            )
    if is_fake_tensor(tensor) or tensor.is_meta:
        # AOT inputs have no real pointer; reuse this contract only when the
        # runtime tensor has the allocator-backed base alignment assumed here.
        return divisibility
    element_bytes = max(dtype.width // 8, 1)
    return min(
        divisibility,
        power_of_2_divisibility(tensor.data_ptr() // element_bytes, max_divisibility),
    )


def register_tensor_epilogue_fn(
    tensor_epilogue_key: str, tensor_epilogue_fn: Callable
) -> None:
    _tensor_epilogue_fns[tensor_epilogue_key] = tensor_epilogue_fn


def register_local_reduce_fns(
    combine_key: str,
    combine_fn: Callable,
    finalize_key: str,
    finalize_fn: Callable,
) -> None:
    _local_reduce_combine_fns[combine_key] = combine_fn
    _local_reduce_finalize_fns[finalize_key] = finalize_fn


class GemmActMixin(ComposableEpiMixin):
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        RowVecTupleLoad("mTensorEpilogueRowVecBroadcasts"),
        ColVecTupleLoad("mTensorEpilogueColVecBroadcasts"),
        TileTupleLoad("mTensorEpilogueTiles"),
        ScalarTupleLoad("mTensorEpilogueScalars"),
        GroupedLocalReduce("mLocalReduce"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("tensor_epilogue_fn", cutlass.Constexpr, None),
        ("tensor_epilogue_arg_kinds", cutlass.Constexpr, ()),
        ("tensor_epilogue_returns_aux", cutlass.Constexpr, False),
        ("tensor_epilogue_returns_local_reduce", cutlass.Constexpr, False),
        ("local_reduce_feeds_main", cutlass.Constexpr, False),
        ("local_reduce_group", cutlass.Constexpr, 0),
        ("local_reduce_axis", cutlass.Constexpr, 1),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        tensor_epilogue_fn: cutlass.Constexpr[Optional[Callable]] = None
        tensor_epilogue_arg_kinds: cutlass.Constexpr[tuple] = ()
        tensor_epilogue_returns_aux: cutlass.Constexpr[bool] = False
        tensor_epilogue_returns_local_reduce: cutlass.Constexpr[bool] = False
        local_reduce_feeds_main: cutlass.Constexpr[bool] = False
        local_reduce_group: cutlass.Constexpr[int] = 0
        local_reduce_axis: cutlass.Constexpr[int] = 1
        local_reduce_combine_fn: cutlass.Constexpr[Optional[Callable]] = None
        local_reduce_finalize_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mTensorEpilogueRowVecBroadcasts: Optional[tuple[cute.Tensor, ...]] = None
        mTensorEpilogueColVecBroadcasts: Optional[tuple[cute.Tensor, ...]] = None
        mTensorEpilogueTiles: Optional[tuple[cute.Tensor, ...]] = None
        mTensorEpilogueScalars: Optional[tuple[cute.Tensor, ...]] = None
        mLocalReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def _filter_epi_ops(self, args):
        super()._filter_epi_ops(args)
        if args.local_reduce_feeds_main and not any(
            op.name == "mLocalReduce" for op in self._epi_ops
        ):
            reduce_ops = tuple(
                op for op in type(self)._epi_ops if op.name == "mLocalReduce"
            )
            self._epi_ops = (*self._epi_ops, *reduce_ops)

    def epi_to_underlying_arguments(self, args: EpilogueArguments, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        if isinstance(args.mAuxOut, tuple):
            self.aux_out_dtypes = tuple(tensor.element_type for tensor in args.mAuxOut)
            self.aux_out_layouts = tuple(
                cutlass.utils.LayoutEnum.from_tensor(tensor) for tensor in args.mAuxOut
            )
        else:
            self.aux_out_dtype = args.mAuxOut.element_type
            self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        d["tensor_epilogue_fn"] = args.tensor_epilogue_fn
        d["tensor_epilogue_arg_kinds"] = args.tensor_epilogue_arg_kinds
        d["tensor_epilogue_returns_aux"] = args.tensor_epilogue_returns_aux
        d["tensor_epilogue_returns_local_reduce"] = args.tensor_epilogue_returns_local_reduce
        d["local_reduce_feeds_main"] = args.local_reduce_feeds_main
        d["local_reduce_group"] = args.local_reduce_group
        d["local_reduce_axis"] = args.local_reduce_axis
        self.local_reduce_feeds_main = args.local_reduce_feeds_main
        self.local_reduce_group = args.local_reduce_group
        self.local_reduce_axis = args.local_reduce_axis
        for key in ("mRowVecBroadcast", "mColVecBroadcast"):
            if key in self.concat_layout and key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        return self.EpilogueParams(**d)

    @classmethod
    def epi_smem_bytes(cls, args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        result = super().epi_smem_bytes(args, cta_tile_shape_mnk, epi_tile, warp_shape_mnk)
        if args.mLocalReduce is not None and grouped_local_reduce_uses_smem(
            args.local_reduce_axis, args.local_reduce_group
        ):
            smem_warps = max(
                (warp_shape_mnk[0] if warp_shape_mnk is not None else 1) - 1, 0
            )
            result += EpiSmemBytes(
                unstaged=cta_tile_shape_mnk[1] * smem_warps * (Float32.width // 8)
            )
        return result

    # epi_get_tma_atoms, epi_get_smem_struct, and epi_get_smem_tensors are all
    # inherited from ComposableEpiMixin via _epi_ops.

    def epi_make_aux_out_copy_atom_r2s(self, params, tiled_copy_t2r, index=None):
        """Build the register-to-shared copy atom used by aux outputs."""
        aux_out_layout = (
            self.aux_out_layouts[index] if index is not None else self.aux_out_layout
        )
        aux_out_dtype = self.aux_out_dtypes[index] if index is not None else self.aux_out_dtype
        epi_tile_mAuxOut = (
            params.epi_tile_mAuxOut[index]
            if index is not None
            else params.epi_tile_mAuxOut
        )
        if self.arch == 100:
            return sm100_utils.get_smem_store_op(
                aux_out_layout, aux_out_dtype, self.acc_dtype, tiled_copy_t2r
            )
        else:
            return copy_utils.get_smem_store_atom(
                aux_out_dtype,
                transpose=aux_out_layout != cutlass.utils.LayoutEnum.ROW_MAJOR,
                major_mode_size=cute.size(epi_tile_mAuxOut, mode=[1])
                // self.atom_layout_mnk[1],
            )

    def epi_make_aux_out_tiled_copy_r2s(
        self, params, tiled_copy_r2s, tiled_copy_t2r, index=None
    ):
        """Build the register-to-shared tiled copy used by aux outputs."""
        copy_atom_aux_out_r2s = self.epi_make_aux_out_copy_atom_r2s(
            params, tiled_copy_t2r, index
        )
        return cute.make_tiled_copy_S(copy_atom_aux_out_r2s, tiled_copy_r2s)

    def epi_setup_aux_out(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Setup aux output TMA copies and partitions before the epilogue loop.

        Returns None when mAuxOut wasn't supplied so the framework skips the aux-out path.
        """
        if getattr(params, "mAuxOut", None) is None:
            return None
        sAuxOut = epi_smem_tensors["mAuxOut"]
        if isinstance(params.mAuxOut, tuple):
            contexts = []
            batch_idx = tile_coord_mnkl[3]
            for i, aux_out in enumerate(params.mAuxOut):
                tiled_copy_aux_out_r2s = self.epi_make_aux_out_tiled_copy_r2s(
                    params, tiled_copy_r2s, tiled_copy_t2r, i
                )
                tRS_sAuxOut = tiled_copy_aux_out_r2s.get_slice(tidx).partition_D(
                    sAuxOut[i]
                )
                copy_aux_out, _, _ = self.epilog_gmem_copy_and_partition(
                    params.tma_atom_mAuxOut[i],
                    varlen_manager.offset_batch_epi(aux_out, batch_idx),
                    self.cta_tile_shape_aux_out_mn,
                    params.epi_tile_mAuxOut[i],
                    sAuxOut[i],
                    tile_coord_mnkl,
                )
                contexts.append((tiled_copy_aux_out_r2s, tRS_sAuxOut, copy_aux_out))
            return tuple(contexts)
        tiled_copy_aux_out_r2s = self.epi_make_aux_out_tiled_copy_r2s(
            params, tiled_copy_r2s, tiled_copy_t2r
        )
        tRS_sAuxOut = tiled_copy_aux_out_r2s.get_slice(tidx).partition_D(sAuxOut)
        batch_idx = tile_coord_mnkl[3]
        copy_aux_out, _, _ = self.epilog_gmem_copy_and_partition(
            params.tma_atom_mAuxOut,
            varlen_manager.offset_batch_epi(params.mAuxOut, batch_idx),
            self.cta_tile_shape_aux_out_mn,
            params.epi_tile_mAuxOut,
            sAuxOut,
            tile_coord_mnkl,
        )
        return ((tiled_copy_aux_out_r2s, tRS_sAuxOut, copy_aux_out),)

    @cute.jit
    def epi_convert_one_aux_out(
        self,
        tRS_rAuxOut,
        aux_out_dtype: cutlass.Constexpr,
        sr_seed,
        tidx,
        tile_coord_mnkl,
        num_prev_subtiles,
        epi_idx,
    ):
        if const_expr(
            self.rounding_mode == RoundingMode.RS
            and tRS_rAuxOut.element_type == cutlass.Float32
            and aux_out_dtype == cutlass.BFloat16
        ):
            from cutlass.cute.tensor import TensorSSA

            seed = epilogue_aux_out_sr_seed(
                sr_seed, tile_coord_mnkl, num_prev_subtiles + epi_idx
            )
            tRS_rAuxOut_out = cute.make_rmem_tensor_like(tRS_rAuxOut, aux_out_dtype)
            src_vec = tRS_rAuxOut.load()
            raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tidx)
            tRS_rAuxOut_out.store(TensorSSA(raw_vec, src_vec.shape, aux_out_dtype))
        else:
            tRS_rAuxOut_out = cute.make_rmem_tensor_like(tRS_rAuxOut, aux_out_dtype)
            tRS_rAuxOut_out.store(tRS_rAuxOut.load().to(aux_out_dtype))
        return tRS_rAuxOut_out

    @cute.jit
    def epi_convert_aux_out(
        self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert aux outputs from acc_dtype to their output dtypes."""
        if const_expr(isinstance(tRS_rAuxOut, tuple)):
            result = []
            for i, aux_out in enumerate(tRS_rAuxOut):
                result.append(
                    self.epi_convert_one_aux_out(
                        aux_out,
                        self.aux_out_dtypes[i],
                        sr_seed,
                        tidx,
                        tile_coord_mnkl,
                        num_prev_subtiles,
                        epi_idx,
                    )
                )
            return tuple(result)
        return self.epi_convert_one_aux_out(
            tRS_rAuxOut,
            self.aux_out_dtype,
            sr_seed,
            tidx,
            tile_coord_mnkl,
            num_prev_subtiles,
            epi_idx,
        )

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        tDrLocalReduceValue = None
        if const_expr(params.local_reduce_feeds_main):
            tDrLocalReduce = epi_loop_tensors.get("mLocalReduce")
            combine_fn = const_expr(params.mLocalReduce.combine_fn)
            finalize_fn = const_expr(params.mLocalReduce.finalize_fn)
            tDrLocalReduceValue = grouped_rowvec_reduce_value(
                self, tRS_rD, tDrLocalReduce, combine_fn, finalize_fn
            )
        if const_expr(params.tensor_epilogue_fn is None or not params.tensor_epilogue_arg_kinds):
            GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(params.tensor_epilogue_fn is not None):
            tRS_rEpilogueIn = cute.make_rmem_tensor_like(tRS_rD, self.acc_dtype)
            tRS_rEpilogueIn.store(tRS_rD.load())
            epilogue_aux_values = []
            if const_expr(params.tensor_epilogue_arg_kinds):
                tDrRowVecs = epi_loop_tensors.get("mTensorEpilogueRowVecBroadcasts")
                tDrColVecs = epi_loop_tensors.get("mTensorEpilogueColVecBroadcasts")
                tRsTileAuxes = epi_loop_tensors.get("mTensorEpilogueTiles")
                scalars = epi_loop_tensors.get("mTensorEpilogueScalars")
                row_arg_index = 0
                col_arg_index = 0
                tile_arg_index = 0
                scalar_arg_index = 0
                for arg_kind in params.tensor_epilogue_arg_kinds:
                    if const_expr(arg_kind == 4):
                        scalar = scalars[scalar_arg_index]
                        scalar_arg_index += 1
                        tRS_rScalar = cute.make_rmem_tensor_like(tRS_rD, scalar.dtype)
                        tRS_rScalar.fill(scalar[0])
                        epilogue_aux_values.append(tRS_rScalar.load())
                    else:
                        if const_expr(arg_kind == 1):
                            epilogue_aux = tRsTileAuxes[tile_arg_index]
                            tile_arg_index += 1
                        elif const_expr(arg_kind == 2):
                            epilogue_aux = tDrRowVecs[row_arg_index]
                            row_arg_index += 1
                        else:
                            epilogue_aux = tDrColVecs[col_arg_index]
                            col_arg_index += 1
                        tRS_rEpilogueAux = cute.make_rmem_tensor_like(
                            tRS_rD, epilogue_aux.element_type
                        )
                        tRS_rEpilogueAux.store(epilogue_aux.load())
                        epilogue_aux_values.append(tRS_rEpilogueAux.load())
            if const_expr(tDrLocalReduceValue is not None):
                epilogue_aux_values.append(tDrLocalReduceValue.load())
            epilogue_result = params.tensor_epilogue_fn(
                tRS_rEpilogueIn.load(), *tuple(epilogue_aux_values)
            )
            if const_expr(params.tensor_epilogue_returns_aux):
                tRS_rD.store(epilogue_result[0])
                aux_results = []
                for i, _ in enumerate(params.mAuxOut):
                    aux_result = epilogue_result[i + 1]
                    tRS_rAuxOut = cute.make_rmem_tensor(
                        aux_result.shape,
                        aux_result.element_type,
                    )
                    tRS_rAuxOut.store(aux_result)
                    aux_results.append(tRS_rAuxOut)
                if const_expr(params.tensor_epilogue_returns_local_reduce):
                    tDrLocalReduce = epi_loop_tensors.get("mLocalReduce")
                    if const_expr(params.local_reduce_feeds_main):
                        tDrLocalReduce = tDrLocalReduce.local_reduce
                    tDrLocalReduce.store(epilogue_result[len(params.mAuxOut) + 1])
                tRS_rAuxOut = tuple(aux_results)
            elif const_expr(params.tensor_epilogue_returns_local_reduce):
                tDrLocalReduce = epi_loop_tensors.get("mLocalReduce")
                if const_expr(params.local_reduce_feeds_main):
                    tDrLocalReduce = tDrLocalReduce.local_reduce
                tRS_rD.store(epilogue_result[0])
                tDrLocalReduce.store(epilogue_result[1])
                tRS_rAuxOut = cute.make_rmem_tensor(
                    epilogue_result[0].shape, self.acc_dtype
                )
                tRS_rAuxOut.store(epilogue_result[0])
            else:
                tRS_rAuxOut = cute.make_rmem_tensor(
                    epilogue_result.shape, self.acc_dtype
                )
                tRS_rAuxOut.store(epilogue_result)
        elif const_expr(params.act_fn is not None):
            tRS_rAuxOut = cute.make_rmem_tensor(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch != 100):
                for i in cutlass.range(cute.size(tRS_rAuxOut), unroll_full=True):
                    tRS_rAuxOut[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                    tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rAuxOut = tRS_rD
        return tRS_rAuxOut


class GemmActSm90(GemmActMixin, GemmSm90):
    pass


class GemmActSm80(GemmActMixin, GemmSm80):
    pass


class GemmActSm100(GemmActMixin, GemmSm100):
    pass


class GemmActSm120(GemmActMixin, GemmSm120):
    pass


def _gated_epi_tile_fn(gemm, epi_tile):
    """Halve the N dimension of the epi_tile for gated postact."""
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(2, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // 2)



class GemmGatedMixin(GemmActMixin):
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        TileStore("mAuxOut", epi_tile_fn=_gated_epi_tile_fn),
    )

    def epi_to_underlying_arguments(
        self, args: GemmActMixin.EpilogueArguments, *, loc=None, ip=None
    ) -> GemmActMixin.EpilogueParams:
        assert args.mAuxOut.element_type.width == 16, (
            "GemmGated only supports 16bit postact for now"
        )
        assert self.d_layout is None or self.d_layout.is_n_major_c()
        assert cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut).is_n_major_c()
        if self.arch == 90:
            assert self.cta_tile_shape_mnk[1] % 32 == 0, (
                "GemmGatedSm90 requires tileN to be divisible by 32"
            )
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2,
        )
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        for key in ("mRowVecBroadcast", "mColVecBroadcast"):
            if key in self.concat_layout and key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(
        self,
        params: GemmActMixin.EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        tRS_rAuxOut_layout = cute.recast_layout(2, 1, tRS_rD.layout)
        # If we don't have .shape here, the compiler generates local stores and loads
        tRS_rAuxOut = cute.make_rmem_tensor(tRS_rAuxOut_layout.shape, self.acc_dtype)
        if const_expr(self.arch != 100):
            for i in cutlass.range(cute.size(tRS_rAuxOut), unroll_full=True):
                tRS_rAuxOut[i] = params.act_fn(tRS_rD[2 * i], tRS_rD[2 * i + 1])
        else:
            for i in cutlass.range(cute.size(tRS_rAuxOut) // 2, unroll_full=True):
                tRS_rAuxOut[2 * i], tRS_rAuxOut[2 * i + 1] = params.act_fn(
                    (tRS_rD[4 * i], tRS_rD[4 * i + 2]), (tRS_rD[4 * i + 1], tRS_rD[4 * i + 3])
                )
        return tRS_rAuxOut

    @cute.jit
    def epi_convert_aux_out(
        self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        tRS_rAuxOut_out = GemmActMixin.epi_convert_aux_out(
            self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(self.arch in (90, 120)):
            # Only need this if we're using STSM
            permute_gated_Cregs_b16(tRS_rAuxOut_out)
        return tRS_rAuxOut_out


class GemmGatedSm90(GemmGatedMixin, GemmSm90):
    pass


class GemmGatedSm80(GemmGatedMixin, GemmSm80):
    pass


class GemmGatedSm100(GemmGatedMixin, GemmSm100):
    pass


class GemmGatedSm120Mixin:
    @staticmethod
    def _compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        if epi_tile_override is not None:
            return epi_tile_override
        # Typically epi_tile is (64, 32) but since we want tile_n = 64 (see below), we might set
        # tile_m = 32 if there's only 2 warps along the M direction.
        tile_m = math.gcd(atom_layout_mnk[0] * 16, cute.size(cta_tile_shape_mnk, mode=[0]))
        atom_n = atom_layout_mnk[1]
        # E.g. if we have 2 warps along N direction, we want each warp to have 32 elems so that
        # postact has 16 elements, which means tile_n should be 64.
        tile_n = math.gcd(atom_n * 8 * 4, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    def epi_make_aux_out_tiled_copy_r2s(self, params, tiled_copy_r2s, tiled_copy_t2r):
        copy_atom_aux_out_r2s = self.epi_make_aux_out_copy_atom_r2s(params, tiled_copy_t2r)
        copy_atom_postact_c = self.epi_make_aux_out_copy_atom_r2s(params, cutlass.Float16)
        op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        permutation_mnk = (
            self.mma_inst_mnk[0] * atom_m,
            self.mma_inst_mnk[1] * atom_n * 2,
            self.mma_inst_mnk[2] * atom_k,
        )
        tiled_mma_gated_postact = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tiled_copy_aux_out_c_atom = cute.make_tiled_copy_C_atom(
            copy_atom_postact_c, tiled_mma_gated_postact
        )
        return cute.make_tiled_copy_S(copy_atom_aux_out_r2s, tiled_copy_aux_out_c_atom)


class GemmGatedSm120(GemmGatedSm120Mixin, GemmGatedMixin, GemmSm120):
    pass


@jit_cache
def _compile_gemm_act(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    postact_dtypes,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_majors,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    tensor_epilogue_key,
    tensor_epilogue_arg_kinds,
    tensor_epilogue_returns_aux,
    tensor_epilogue_returns_local_reduce,
    local_reduce_feeds_main,
    tensor_epilogue_rowvec_dtypes,
    tensor_epilogue_colvec_dtypes,
    tensor_epilogue_colvec_ndims,
    tensor_epilogue_tile_dtypes,
    tensor_epilogue_tile_majors,
    tensor_epilogue_scalar_dtypes,
    local_reduce_dtype,
    local_reduce_ndim,
    local_reduce_group,
    local_reduce_axis,
    local_reduce_stride_divisibility,
    local_reduce_combine_key,
    local_reduce_finalize_key,
    alpha_mode,
    beta_mode,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    varlen_m,
    gather_A,
    concat_layout,
    device_capacity,
    gemm_cls_name,
    rounding_mode=RoundingMode.RN,
    sr_seed_mode=0,
    use_tma_gather=False,
):
    sm_to_cls = {
        "act": {
            8: GemmActSm80,
            9: GemmActSm90,
            10: GemmActSm100,
            11: GemmActSm100,
            12: GemmActSm120,
        },
        "gated": {
            8: GemmGatedSm80,
            9: GemmGatedSm90,
            10: GemmGatedSm100,
            11: GemmGatedSm100,
            12: GemmGatedSm120,
        },
    }
    GemmCls = sm_to_cls[gemm_cls_name][device_capacity[0]]
    postact_dtype = postact_dtypes[0]
    postact_major = postact_majors[0]
    pa_leading = 1 if postact_major == "n" else 0
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        varlen_m=varlen_m,
        gather_A=gather_A,
    )
    pa_n = cute.sym_int() if gemm_cls_name == "gated" else n
    pa_shape = (m, pa_n) if varlen_m else (m, pa_n, l)
    if tensor_epilogue_returns_aux:
        mAuxOut = tuple(
            fake_tensor(
                dtype,
                pa_shape,
                leading_dim=1 if major == "n" else 0,
                divisibility=div_for_dtype(dtype),
            )
            for dtype, major in zip(postact_dtypes, postact_majors)
        )
    else:
        div_pa = div_for_dtype(postact_dtype)
        pa_leading_dim = 1 if gemm_cls_name == "gated" else pa_leading
        mAuxOut = fake_tensor(
            postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa
        )

    mRowVec = fake_tensor(rowvec_dtype, (l, n), leading_dim=1, divisibility=4)
    mTensorEpilogueRowVecs = tuple(
        fake_tensor(dtype, (l, n), leading_dim=1, divisibility=4)
        for dtype in tensor_epilogue_rowvec_dtypes
    ) or None
    mTensorEpilogueTiles = tuple(
        fake_tensor(
            dtype,
            (m, n, l),
            leading_dim=1 if major == "n" else 0,
            divisibility=div_for_dtype(dtype),
        )
        for dtype, major in zip(tensor_epilogue_tile_dtypes, tensor_epilogue_tile_majors)
    ) or None
    if colvec_ndim == 2:
        mColVec = fake_tensor(colvec_dtype, (l, m), leading_dim=1, divisibility=4)
    elif colvec_ndim == 1:
        mColVec = fake_tensor(colvec_dtype, (m,), leading_dim=0, divisibility=4)
    else:
        mColVec = None
    mTensorEpilogueColVecs = tuple(
        fake_tensor(
            dtype,
            (l, m) if ndim == 2 else (m,),
            leading_dim=1 if ndim == 2 else 0,
            divisibility=4,
        )
        for dtype, ndim in zip(tensor_epilogue_colvec_dtypes, tensor_epilogue_colvec_ndims)
    ) or None
    mTensorEpilogueScalars = tuple(
        fake_tensor(dtype, (1,), leading_dim=0, divisibility=1)
        for dtype in tensor_epilogue_scalar_dtypes
    ) or None
    if local_reduce_dtype is not None and local_reduce_axis == 0:
        local_reduce_shape = (
            (l, cute.sym_int(), n) if local_reduce_ndim == 3 else (cute.sym_int(), n)
        )
    elif local_reduce_dtype is not None:
        local_reduce_shape = (
            (l, m, cute.sym_int()) if local_reduce_ndim == 3 else (m, cute.sym_int())
        )
    else:
        local_reduce_shape = None
    if (
        local_reduce_dtype is None
        or (local_reduce_feeds_main and local_reduce_ndim == 0)
    ):
        mLocalReduce = None
    else:
        local_reduce_leading_dim = 2 if local_reduce_ndim == 3 else 1
        mLocalReduce = fake_tensor(
            local_reduce_dtype,
            local_reduce_shape,
            leading_dim=local_reduce_leading_dim,
            divisibility=local_reduce_stride_divisibility,
        )

    tensor_epilogue_fn = (
        _tensor_epilogue_fns[tensor_epilogue_key]
        if tensor_epilogue_key is not None and activation is None
        else None
    )
    local_reduce_combine_fn = (
        _local_reduce_combine_fns[local_reduce_combine_key]
        if local_reduce_combine_key is not None
        else None
    )
    local_reduce_finalize_fn = (
        _local_reduce_finalize_fns[local_reduce_finalize_key]
        if local_reduce_finalize_key is not None
        else None
    )
    act_fn = None if tensor_epilogue_fn is not None else (
        act_fn_map[activation] if gemm_cls_name == "act" else gate_fn_map[activation]
    )

    def fake_scalar(mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(0)
        else:
            return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        mAuxOut=mAuxOut,
        act_fn=act_fn,
        tensor_epilogue_fn=tensor_epilogue_fn,
        tensor_epilogue_arg_kinds=tensor_epilogue_arg_kinds,
        tensor_epilogue_returns_aux=tensor_epilogue_returns_aux,
        tensor_epilogue_returns_local_reduce=tensor_epilogue_returns_local_reduce,
        local_reduce_feeds_main=local_reduce_feeds_main,
        local_reduce_group=local_reduce_group,
        local_reduce_axis=local_reduce_axis,
        local_reduce_combine_fn=local_reduce_combine_fn,
        local_reduce_finalize_fn=local_reduce_finalize_fn,
        alpha=fake_scalar(alpha_mode, Float32),
        beta=fake_scalar(beta_mode, Float32),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        mTensorEpilogueRowVecBroadcasts=mTensorEpilogueRowVecs,
        mTensorEpilogueColVecBroadcasts=mTensorEpilogueColVecs,
        mTensorEpilogueTiles=mTensorEpilogueTiles,
        mTensorEpilogueScalars=mTensorEpilogueScalars,
        mLocalReduce=mLocalReduce,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(
        varlen_m, False, gather_A, m if varlen_m else None
    )
    return compile_gemm_kernel(
        GemmCls,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        gather_A,
        is_dynamic_persistent,
        device_capacity,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        use_tma_gather=use_tma_gather,
        concat_layout=concat_layout or None,
    )


def gemm_act(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor | tuple[Tensor, ...],  # tensor epilogues pass same-shape aux outputs as a tuple
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: tuple | None = None,
    tensor_epilogue_fn: Optional[Callable] = None,
    tensor_epilogue_key: Optional[str] = None,
    tensor_epilogue_returns_aux: bool = False,
    tensor_epilogue_returns_local_reduce: bool = False,
    local_reduce_feeds_main: bool = False,
    tensor_epilogue_arg_kinds: tuple[str, ...] = (),
    tensor_epilogue_rowvec_biases: tuple[Tensor, ...] = (),
    tensor_epilogue_colvec_biases: tuple[Tensor, ...] = (),
    tensor_epilogue_tile_biases: tuple[Tensor, ...] = (),
    tensor_epilogue_scalar_biases: tuple[Tensor, ...] = (),
    local_reduce_out: Optional[Tensor] = None,
    local_reduce_group: int = 0,
    local_reduce_axis: int = 1,
    local_reduce_combine_key: Optional[str] = None,
    local_reduce_finalize_key: Optional[str] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    device_capacity_override: tuple[int, int] | None = None,
) -> None:
    """Run GEMM with an optional generated tensor epilogue and local reduction.

    Local reductions group the logical GEMM output ``[L, M, N]``. Axis 0 groups
    M and writes ``local_reduce_out`` with shape ``[L, M / group, N]``; axis 1
    groups N and writes ``[L, M, N / group]``. ``local_reduce_group`` must be
    positive, divide the selected dimension, and fit the selected GEMM tile.
    ``local_reduce_out`` is always 3-D at this QuACK boundary.

    When ``tensor_epilogue_returns_local_reduce`` is true, the generated tensor
    epilogue's final return value, after the main result and any same-shape aux
    results, supplies the grouped reduction. The flag and ``local_reduce_out``
    must be provided together, and a tensor epilogue function or registry key is
    required.

    ``local_reduce_feeds_main`` instead reduces the accumulator, broadcasts the
    finalized group value to its member rows, and appends it as the final input
    to the tensor epilogue. It does not request an output store by itself. This
    mode requires a tensor epilogue and both callback keys, and currently accepts
    only axis-0 groups that fit within one warp. The callback omission described
    below applies only to store-only reductions.

    ``local_reduce_combine_key`` and ``local_reduce_finalize_key`` identify
    CuTeDSL callbacks registered with ``register_local_reduce_fns``. The binary
    combine callback merges two physical partial values using the reduction
    operation. The unary finalize callback transforms the fully combined value
    before conversion and storage. Both keys are required whenever the grouped
    reduction needs physical lane, warp, or fragment combining; they may be
    omitted when the tensor epilogue returns an already-complete TensorSSA group.

    Returns:
        None. Results are written to the supplied output tensors.
    """
    if tensor_epilogue_fn is not None:
        assert activation is None, "tensor_epilogue_fn and activation are mutually exclusive"
        tensor_epilogue_key = (
            tensor_epilogue_key
            if tensor_epilogue_key is not None
            else repr(tensor_epilogue_fn)
        )
        register_tensor_epilogue_fn(tensor_epilogue_key, tensor_epilogue_fn)
        gemm_cls_name = "act"
    elif activation in gate_fn_map:
        gemm_cls_name = "gated"
    else:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
        gemm_cls_name = "act"

    if tensor_epilogue_returns_aux:
        if not isinstance(PostAct, tuple):
            raise RuntimeError("tensor epilogue aux outputs must be passed as a tuple")
        postact_tensors = PostAct
    else:
        postact_tensors = (PostAct,)
    varlen_m = cu_seqlens_m is not None
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert all(
            tensor.stride(-1) == 1 for tensor in postact_tensors
        ), "varlen_m requires PostAct to be n-major"
    if gather_A:
        assert varlen_m, "gather_A requires varlen_m"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    A_p = perm3d_single(A, varlen_m)
    B_p = perm3d_single(B)
    D_p = perm3d_single(D, varlen_m)
    C_p = perm3d_single(C, varlen_m)
    if tensor_epilogue_returns_aux:
        PostAct_p = tuple(perm3d_single(tensor, varlen_m) for tensor in postact_tensors)
    else:
        PostAct_p = perm3d_single(PostAct, varlen_m)
    tensor_epilogue_tile_biases_p = tuple(
        perm3d_single(tensor, varlen_m) for tensor in tensor_epilogue_tile_biases
    )

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_majors = (
        tuple(get_major(tensor, "m", "n") for tensor in PostAct_p)
        if isinstance(PostAct_p, tuple)
        else (get_major(PostAct_p, "m", "n"),)
    )

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtypes = (
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in PostAct)
        if isinstance(PostAct, tuple)
        else (torch2cute_dtype_map[PostAct.dtype],)
    )
    if not set(tensor_epilogue_arg_kinds) <= {"tile", "row", "col", "scalar"}:
        raise NotImplementedError(
            f"QUACK tensor epilogues support only tile/row/col/scalar aux tensors, got {tensor_epilogue_arg_kinds}"
        )
    expected_tensor_epilogue_arg_counts = {
        "row": len(tensor_epilogue_rowvec_biases),
        "col": len(tensor_epilogue_colvec_biases),
        "tile": len(tensor_epilogue_tile_biases),
        "scalar": len(tensor_epilogue_scalar_biases),
    }
    actual_tensor_epilogue_arg_counts = {
        kind: tensor_epilogue_arg_kinds.count(kind)
        for kind in expected_tensor_epilogue_arg_counts
    }
    if actual_tensor_epilogue_arg_counts != expected_tensor_epilogue_arg_counts:
        raise RuntimeError(
            "tensor_epilogue_arg_kinds must match row/col/tile/scalar tensor epilogue args"
        )
    tensor_epilogue_arg_kind_codes = tuple(
        {"tile": 1, "row": 2, "col": 3, "scalar": 4}[kind]
        for kind in tensor_epilogue_arg_kinds
    )
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0

    device_capacity = (
        device_capacity_override
        if device_capacity_override is not None
        else get_device_capacity(A.device)
    )
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    if rounding_mode == RoundingMode.RS:
        assert device_capacity[0] == 10, "Stochastic rounding (RoundingMode.RS) requires SM100"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    sr_seed_mode = (
        2 if isinstance(sr_seed, Tensor) else (1 if rounding_mode == RoundingMode.RS else 0)
    )
    alpha_mode = 2 if isinstance(alpha, Tensor) else (1 if alpha != 1.0 else 0)
    beta_mode = 2 if isinstance(beta, Tensor) else (1 if beta != 1.0 else 0)
    if (tensor_epilogue_arg_kinds or local_reduce_feeds_main) and (
        C is not None or alpha_mode != 0 or beta_mode != 0
    ):
        raise NotImplementedError(
            "QUACK tensor epilogues with aux args or feed-main local reductions "
            "cannot be combined with C/alpha/beta yet"
        )
    if tensor_epilogue_returns_local_reduce != (local_reduce_out is not None):
        raise RuntimeError(
            "tensor_epilogue_returns_local_reduce requires local_reduce_out and vice versa"
        )
    if local_reduce_feeds_main:
        if local_reduce_axis != 0:
            raise NotImplementedError("local_reduce_feeds_main currently supports only axis 0")
        if local_reduce_group <= 0:
            raise RuntimeError("local_reduce_group must be positive")
        if local_reduce_group > LOCAL_REDUCE_FRAGMENT_WIDTH:
            raise NotImplementedError(
                "local_reduce_feeds_main currently supports only same-warp axis-0 "
                f"groups <= {LOCAL_REDUCE_FRAGMENT_WIDTH}"
            )
        if tensor_epilogue_fn is None and tensor_epilogue_key is None:
            raise RuntimeError(
                "local_reduce_feeds_main requires tensor_epilogue_fn or tensor_epilogue_key"
            )
        if local_reduce_combine_key is None or local_reduce_finalize_key is None:
            raise RuntimeError(
                "local_reduce_feeds_main requires generated local-reduce callback keys"
            )
        if LOCAL_REDUCE_FRAGMENT_WIDTH % local_reduce_group != 0:
            raise RuntimeError(
                "local_reduce_group must divide TensorSSA fragment width 32"
            )
        if A.shape[-2] % local_reduce_group != 0:
            raise RuntimeError(
                "local_reduce_group must divide the selected GEMM output dimension"
            )
    if local_reduce_out is not None and tensor_epilogue_fn is None and tensor_epilogue_key is None:
        raise RuntimeError("local_reduce_out requires tensor_epilogue_fn")
    if local_reduce_out is not None:
        if local_reduce_group <= 0:
            raise RuntimeError("local_reduce_group must be positive")
        if local_reduce_axis not in (0, 1):
            raise RuntimeError("local_reduce_axis must be 0 or 1")
        if (
            local_reduce_axis == 0
            or local_reduce_group > LOCAL_REDUCE_FRAGMENT_WIDTH
        ) and (
            local_reduce_combine_key is None or local_reduce_finalize_key is None
        ):
            raise RuntimeError(
                "physical local reductions require generated local-reduce callback keys"
            )
        if local_reduce_out.ndim != 3:
            raise NotImplementedError("QUACK local_reduce_out must be 3-D")
    concat_layout = tuple(sorted(concat_layout)) if concat_layout else ()
    local_reduce_dtype = (
        torch2cute_dtype_map[local_reduce_out.dtype]
        if local_reduce_out is not None
        else (Float32 if local_reduce_feeds_main else None)
    )
    local_reduce_leading_dim = (
        2 if local_reduce_out is not None and local_reduce_out.ndim == 3 else 1
    )
    local_reduce_stride_divisibility = (
        tensor_stride_divisibility(
            local_reduce_out, local_reduce_dtype, local_reduce_leading_dim
        )
        if local_reduce_out is not None
        else 1
    )
    compiled_fn = _compile_gemm_act(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtypes,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_majors,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        tensor_epilogue_key,
        tensor_epilogue_arg_kind_codes,
        tensor_epilogue_returns_aux,
        tensor_epilogue_returns_local_reduce,
        local_reduce_feeds_main,
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_rowvec_biases),
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_colvec_biases),
        tuple(tensor.ndim for tensor in tensor_epilogue_colvec_biases),
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_tile_biases_p),
        tuple(get_major(tensor, "m", "n") for tensor in tensor_epilogue_tile_biases_p),
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_scalar_biases),
        local_reduce_dtype,
        local_reduce_out.ndim if local_reduce_out is not None else 0,
        local_reduce_group,
        local_reduce_axis,
        local_reduce_stride_divisibility,
        local_reduce_combine_key,
        local_reduce_finalize_key,
        alpha_mode,
        beta_mode,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        varlen_m,
        gather_A,
        concat_layout,
        device_capacity,
        gemm_cls_name,
        rounding_mode=rounding_mode,
        sr_seed_mode=sr_seed_mode,
        use_tma_gather=use_tma_gather,
    )

    from .cache import is_compile_only

    if is_compile_only():
        return

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0

    def scalar_arg(scalar, mode, dtype=Int32):
        if mode == 0:
            return None
        elif mode == 1:
            return dtype(scalar)
        else:
            return scalar.data_ptr()

    epi_args = GemmActMixin.EpilogueArguments(
        mAuxOut=PostAct_p,
        act_fn=None,
        tensor_epilogue_fn=None,
        tensor_epilogue_arg_kinds=None,
        tensor_epilogue_returns_aux=None,
        tensor_epilogue_returns_local_reduce=None,
        local_reduce_feeds_main=None,
        local_reduce_group=None,
        local_reduce_axis=None,
        local_reduce_combine_fn=None,
        local_reduce_finalize_fn=None,
        alpha=scalar_arg(alpha, alpha_mode, Float32),
        beta=scalar_arg(beta, beta_mode, Float32),
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        mTensorEpilogueRowVecBroadcasts=tensor_epilogue_rowvec_biases or None,
        mTensorEpilogueColVecBroadcasts=tensor_epilogue_colvec_biases or None,
        mTensorEpilogueTiles=tensor_epilogue_tile_biases_p or None,
        mTensorEpilogueScalars=tensor_epilogue_scalar_biases or None,
        mLocalReduce=local_reduce_out,
        rounding_mode=None,  # Constexpr, pass None at call time
        sr_seed=scalar_arg(sr_seed, sr_seed_mode),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(cu_seqlens_m, None, A_idx)

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)


gemm_gated = gemm_act
