# Copyright (c) 2025, Wentao Guo, Tri Dao.
from __future__ import annotations
import math
from typing import NamedTuple, Tuple, Optional, Callable, Type

from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Float32, const_expr
from cutlass.cute.runtime import make_ptr
from cutlass.cute.nvgpu import warp

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import (
    ensure_varlen_n_supported,
    mlir_namedtuple,
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from .epi_composable import ComposableEpiMixin
from .epi_ops import (
    ColVecLoad,
    GroupedColVecReduce,
    GroupedRowVecReduce,
    RowVecLoad,
    RowVecTupleLoad,
    ColVecTupleLoad,
    Scalar,
    TileLoad,
    TileTupleLoad,
    TileStore,
    colvec_reduce_accumulate,
    grouped_colvec_reduce_accumulate,
    grouped_colvec_reduce_accumulate_amax_abs,
    grouped_rowvec_reduce_accumulate,
    grouped_rowvec_reduce_value,
)
from .gemm_sm80 import GemmSm80
from .gemm_sm90 import GemmSm90
from .gemm_sm100 import GemmSm100
from .gemm_sm120 import GemmSm120
from .gemm_default_epi import GemmDefaultEpiMixin
from .gemm_tvm_ffi_utils import (
    get_major,
    perm3d,
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
        GroupedColVecReduce("mColVecReduce"),
        GroupedRowVecReduce("mRowVecReduce"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = (
        ("act_fn", cutlass.Constexpr, None),
        ("tensor_epilogue_fn", cutlass.Constexpr, None),
        ("tensor_epilogue_uses_c", cutlass.Constexpr, False),
        ("tensor_epilogue_returns_aux", cutlass.Constexpr, False),
        ("tensor_epilogue_arg_kinds", cutlass.Constexpr, ()),
        ("local_reduce_feeds_main", cutlass.Constexpr, False),
        ("local_reduce_source_from_epilogue", cutlass.Constexpr, False),
        ("local_reduce_group", cutlass.Constexpr, 0),
        ("local_reduce_dim", cutlass.Constexpr, 1),
        ("local_reduce_op", cutlass.Constexpr, 0),
        ("local_reduce_scale", cutlass.Constexpr, 1.0),
        ("local_reduce_max_power", cutlass.Constexpr, 8),
    )

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mAuxOut: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        tensor_epilogue_fn: cutlass.Constexpr[Optional[Callable]] = None
        tensor_epilogue_uses_c: cutlass.Constexpr[bool] = False
        tensor_epilogue_returns_aux: cutlass.Constexpr[bool] = False
        tensor_epilogue_arg_kinds: cutlass.Constexpr[tuple] = ()
        local_reduce_feeds_main: cutlass.Constexpr[bool] = False
        local_reduce_source_from_epilogue: cutlass.Constexpr[bool] = False
        local_reduce_group: cutlass.Constexpr[int] = 0
        local_reduce_dim: cutlass.Constexpr[int] = 1
        local_reduce_op: cutlass.Constexpr[int] = 0
        local_reduce_scale: cutlass.Constexpr[float] = 1.0
        local_reduce_max_power: cutlass.Constexpr[int] = 8
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mTensorEpilogueRowVecBroadcasts: Optional[tuple[cute.Tensor, ...]] = None
        mTensorEpilogueColVecBroadcasts: Optional[tuple[cute.Tensor, ...]] = None
        mTensorEpilogueTiles: Optional[tuple[cute.Tensor, ...]] = None
        mColVecReduce: Optional[cute.Tensor] = None
        mRowVecReduce: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops + _extra_param_fields

    def epi_to_underlying_arguments(self, args: EpilogueArguments, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        d["act_fn"] = args.act_fn
        d["tensor_epilogue_fn"] = args.tensor_epilogue_fn
        d["tensor_epilogue_uses_c"] = args.tensor_epilogue_uses_c
        d["tensor_epilogue_returns_aux"] = args.tensor_epilogue_returns_aux
        d["tensor_epilogue_arg_kinds"] = args.tensor_epilogue_arg_kinds
        d["local_reduce_feeds_main"] = args.local_reduce_feeds_main
        d["local_reduce_source_from_epilogue"] = args.local_reduce_source_from_epilogue
        d["local_reduce_group"] = args.local_reduce_group
        d["local_reduce_dim"] = args.local_reduce_dim
        d["local_reduce_op"] = args.local_reduce_op
        d["local_reduce_scale"] = args.local_reduce_scale
        d["local_reduce_max_power"] = args.local_reduce_max_power
        self.local_reduce_feeds_main = args.local_reduce_feeds_main
        self.local_reduce_source_from_epilogue = args.local_reduce_source_from_epilogue
        self.local_reduce_group = args.local_reduce_group
        self.local_reduce_dim = args.local_reduce_dim
        self.local_reduce_op = args.local_reduce_op
        self.local_reduce_scale = args.local_reduce_scale
        self.local_reduce_max_power = args.local_reduce_max_power
        for key in ("mRowVecBroadcast", "mColVecBroadcast"):
            if key in self.concat_layout and key in d:
                d[key] = layout_utils.concat_to_interleave(d[key], 1)
        return self.EpilogueParams(**d)

    # epi_get_tma_atoms, epi_smem_bytes, epi_get_smem_struct,
    # epi_get_smem_tensors are all inherited from ComposableEpiMixin via _epi_ops.

    def epi_make_aux_out_copy_atom_r2s(self, params, tiled_copy_t2r):
        """Build the register-to-shared copy atom used by aux outputs."""
        if self.arch == 100:
            return sm100_utils.get_smem_store_op(
                self.aux_out_layout, self.aux_out_dtype, self.acc_dtype, tiled_copy_t2r
            )
        else:
            return copy_utils.get_smem_store_atom(
                self.aux_out_dtype,
                transpose=self.aux_out_layout != cutlass.utils.LayoutEnum.ROW_MAJOR,
                major_mode_size=cute.size(params.epi_tile_mAuxOut, mode=[1])
                // self.atom_layout_mnk[1],
            )

    def epi_make_aux_out_tiled_copy_r2s(self, params, tiled_copy_r2s, tiled_copy_t2r):
        """Build the register-to-shared tiled copy used by aux outputs."""
        copy_atom_aux_out_r2s = self.epi_make_aux_out_copy_atom_r2s(params, tiled_copy_t2r)
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
        return tiled_copy_aux_out_r2s, tRS_sAuxOut, copy_aux_out

    @cute.jit
    def epi_convert_aux_out(
        self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        """Convert aux output from acc_dtype to aux_out_dtype. Override for custom postprocessing."""
        if const_expr(
            self.rounding_mode == RoundingMode.RS
            and tRS_rAuxOut.element_type == cutlass.Float32
            and self.aux_out_dtype == cutlass.BFloat16
        ):
            from cutlass.cute.tensor import TensorSSA

            seed = epilogue_aux_out_sr_seed(sr_seed, tile_coord_mnkl, num_prev_subtiles + epi_idx)
            tRS_rAuxOut_out = cute.make_rmem_tensor_like(tRS_rAuxOut, self.aux_out_dtype)
            src_vec = tRS_rAuxOut.load()
            raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tidx)
            tRS_rAuxOut_out.store(TensorSSA(raw_vec, src_vec.shape, self.aux_out_dtype))
        else:
            tRS_rAuxOut_out = cute.make_rmem_tensor_like(tRS_rAuxOut, self.aux_out_dtype)
            tRS_rAuxOut_out.store(tRS_rAuxOut.load().to(self.aux_out_dtype))
        return tRS_rAuxOut_out

    @cute.jit
    def epi_visit_subtile(
        self,
        params,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        tDrColVecReduce = epi_loop_tensors.get("mColVecReduce")
        tDrRowVecReduce = epi_loop_tensors.get("mRowVecReduce")
        if const_expr(tDrRowVecReduce is not None and not params.local_reduce_source_from_epilogue):
            if const_expr(params.local_reduce_feeds_main and params.local_reduce_dim == 0):
                tDrRowVecReduceVal = grouped_rowvec_reduce_value(self, tRS_rD, tDrRowVecReduce)
                for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                    tRS_rD[i] /= tDrRowVecReduceVal[i]
            else:
                grouped_rowvec_reduce_accumulate(self, tDrRowVecReduce, tRS_rD)
        if const_expr(tDrColVecReduce is not None and not params.local_reduce_source_from_epilogue):
            if const_expr(params.local_reduce_group != 0 and params.local_reduce_group < self.cta_tile_shape_mnk[1]):
                if const_expr(
                    params.local_reduce_op == 1
                    or params.local_reduce_op == 2
                    or params.local_reduce_op == 3
                ):
                    grouped_colvec_reduce_accumulate_amax_abs(
                        self, tDrColVecReduce, tRS_rD
                    )
                else:
                    grouped_colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD)
            else:
                if const_expr(
                    params.local_reduce_op == 1
                    or params.local_reduce_op == 2
                    or params.local_reduce_op == 3
                ):
                    colvec_reduce_accumulate(
                        self,
                        tDrColVecReduce,
                        tRS_rD,
                        transform_fn=lambda x: cute.arch.fmax(x, -x),
                    )
                else:
                    colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD)
            if const_expr(params.local_reduce_feeds_main):
                if const_expr(self.arch != 100):
                    for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                        tRS_rD[i] /= tDrColVecReduce[i]
                else:
                    for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                        tRS_rD[i] /= tDrColVecReduce[i]
        if const_expr(params.tensor_epilogue_fn is None or not params.tensor_epilogue_uses_c):
            GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        if const_expr(params.tensor_epilogue_fn is not None):
            tRS_rEpilogueIn = cute.make_rmem_tensor_like(tRS_rD, self.acc_dtype)
            tRS_rEpilogueIn.store(tRS_rD.load())
            if const_expr(params.tensor_epilogue_uses_c):
                tDrRowVecs = epi_loop_tensors.get("mTensorEpilogueRowVecBroadcasts")
                tDrColVecs = epi_loop_tensors.get("mTensorEpilogueColVecBroadcasts")
                tRsTileAuxes = epi_loop_tensors.get("mTensorEpilogueTiles")
                epilogue_aux_values = []
                row_arg_index = 0
                col_arg_index = 0
                tile_arg_index = 0
                for arg_kind in params.tensor_epilogue_arg_kinds:
                    tRS_rEpilogueAux = cute.make_rmem_tensor_like(tRS_rD, self.acc_dtype)
                    if const_expr(arg_kind == 1):
                        tRS_rEpilogueAux.store(tRsTileAuxes[tile_arg_index].load().to(self.acc_dtype))
                        tile_arg_index += 1
                    elif const_expr(arg_kind == 2):
                        tRS_rEpilogueAux.store(tDrRowVecs[row_arg_index].load().to(self.acc_dtype))
                        row_arg_index += 1
                    else:
                        tRS_rEpilogueAux.store(tDrColVecs[col_arg_index].load().to(self.acc_dtype))
                        col_arg_index += 1
                    epilogue_aux_values.append(tRS_rEpilogueAux.load())
                epilogue_result = params.tensor_epilogue_fn(
                    tRS_rEpilogueIn.load(), *tuple(epilogue_aux_values)
                )
            else:
                epilogue_result = params.tensor_epilogue_fn(tRS_rEpilogueIn.load())
            if const_expr(params.local_reduce_source_from_epilogue):
                tRS_rD.store(epilogue_result[0])
                tRS_rLocalReduceSource = cute.make_rmem_tensor(
                    epilogue_result[1].shape, self.acc_dtype
                )
                tRS_rLocalReduceSource.store(epilogue_result[1])
                if const_expr(tDrColVecReduce is not None):
                    if const_expr(params.local_reduce_group != 0 and params.local_reduce_group < self.cta_tile_shape_mnk[1]):
                        if const_expr(
                            params.local_reduce_op == 1
                            or params.local_reduce_op == 2
                            or params.local_reduce_op == 3
                        ):
                            grouped_colvec_reduce_accumulate_amax_abs(
                                self, tDrColVecReduce, tRS_rLocalReduceSource
                            )
                        else:
                            grouped_colvec_reduce_accumulate(
                                self, tDrColVecReduce, tRS_rLocalReduceSource
                            )
                    else:
                        if const_expr(
                            params.local_reduce_op == 1
                            or params.local_reduce_op == 2
                            or params.local_reduce_op == 3
                        ):
                            colvec_reduce_accumulate(
                                self,
                                tDrColVecReduce,
                                tRS_rLocalReduceSource,
                                transform_fn=lambda x: cute.arch.fmax(x, -x),
                            )
                        else:
                            colvec_reduce_accumulate(
                                self, tDrColVecReduce, tRS_rLocalReduceSource
                            )
                tRS_rAuxOut = cute.make_rmem_tensor(
                    epilogue_result[0].shape, self.acc_dtype
                )
                tRS_rAuxOut.store(epilogue_result[0])
            elif const_expr(params.tensor_epilogue_returns_aux):
                tRS_rD.store(epilogue_result[0])
                tRS_rAuxOut = cute.make_rmem_tensor(
                    epilogue_result[1].shape, self.acc_dtype
                )
                tRS_rAuxOut.store(epilogue_result[1])
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


def _grouped_n_contract_epi_tile(epi_tile, group):
    if isinstance(epi_tile[1], cute.Layout):
        return (epi_tile[0], cute.recast_layout(group, 1, epi_tile[1]))
    return (epi_tile[0], epi_tile[1] // group)


def _grouped_n_contract_epi_tile_fn(gemm, epi_tile):
    return _grouped_n_contract_epi_tile(epi_tile, 2)


def _grouped_n_contract4_epi_tile_fn(gemm, epi_tile):
    return _grouped_n_contract_epi_tile(epi_tile, 4)


class GemmGroupedNContractMixin(GemmActMixin):
    grouped_n_contract_group = 2
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        GroupedColVecReduce("mColVecReduce"),
        GroupedRowVecReduce("mRowVecReduce"),
        TileStore("mAuxOut", epi_tile_fn=_grouped_n_contract_epi_tile_fn),
    )

    def epi_to_underlying_arguments(
        self, args: GemmActMixin.EpilogueArguments, *, loc=None, ip=None
    ):
        if self.grouped_n_contract_group != 2 and self.arch != 100:
            raise NotImplementedError(
                "grouped_n_contract groups larger than 2 are currently validated only on SM100"
            )
        params = super().epi_to_underlying_arguments(args, loc=loc, ip=ip)
        self.cta_tile_shape_aux_out_mn = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // self.grouped_n_contract_group,
        )
        return params

    @cute.jit
    def epi_convert_aux_out(
        self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
    ):
        tRS_rAuxOut_out = GemmActMixin.epi_convert_aux_out(
            self, tRS_rAuxOut, sr_seed, tidx, tile_coord_mnkl, num_prev_subtiles, epi_idx
        )
        if const_expr(
            self.grouped_n_contract_group == 2
            and self.arch in (90, 120)
            and self.aux_out_dtype.width == 16
        ):
            # Half-N contracted stores use the same b16 register permutation as gated stores.
            permute_gated_Cregs_b16(tRS_rAuxOut_out)
        return tRS_rAuxOut_out


class GemmGroupedNContractSm80(GemmGroupedNContractMixin, GemmSm80):
    pass


class GemmGroupedNContractSm90(GemmGroupedNContractMixin, GemmSm90):
    pass


class GemmGroupedNContractSm100(GemmGroupedNContractMixin, GemmSm100):
    pass


class GemmGroupedNContractSm120(GemmGroupedNContractMixin, GemmSm120):
    pass


class GemmGroupedNContract4Sm100(GemmGroupedNContractMixin, GemmSm100):
    grouped_n_contract_group = 4
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        GroupedColVecReduce("mColVecReduce"),
        GroupedRowVecReduce("mRowVecReduce"),
        TileStore("mAuxOut", epi_tile_fn=_grouped_n_contract4_epi_tile_fn),
    )


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
    postact_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    postact_major,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    is_dynamic_persistent,
    activation,
    tensor_epilogue_fn,
    tensor_epilogue_key,
    tensor_epilogue_uses_c,
    tensor_epilogue_returns_aux,
    tensor_epilogue_arg_kinds,
    tensor_epilogue_rowvec_dtypes,
    tensor_epilogue_colvec_dtypes,
    tensor_epilogue_colvec_ndims,
    tensor_epilogue_tile_dtypes,
    tensor_epilogue_tile_majors,
    alpha_mode,
    beta_mode,
    rowvec_dtype,
    colvec_dtype,
    colvec_ndim,
    local_reduce_dtype,
    local_reduce_ndim,
    local_reduce_feeds_main,
    local_reduce_source_from_epilogue,
    local_reduce_group,
    local_reduce_dim,
    local_reduce_op,
    local_reduce_scale,
    local_reduce_max_power,
    main_output_transform_group,
    varlen_m,
    varlen_k,
    varlen_n,
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
        "grouped_n_contract": {
            8: GemmGroupedNContractSm80,
            9: GemmGroupedNContractSm90,
            10: GemmGroupedNContractSm100,
            11: GemmGroupedNContractSm100,
            12: GemmGroupedNContractSm120,
        },
    }
    GemmCls = sm_to_cls[gemm_cls_name][device_capacity[0]]
    if gemm_cls_name == "grouped_n_contract" and main_output_transform_group != 2:
        if device_capacity[0] != 10:
            raise NotImplementedError(
                "grouped_n_contract groups larger than 2 are currently validated only on SM100"
            )
        GemmCls = {4: GemmGroupedNContract4Sm100}.get(main_output_transform_group)
        if GemmCls is None:
            raise NotImplementedError(
                f"unsupported grouped_n_contract group={main_output_transform_group}"
            )
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
        varlen_k=varlen_k,
        varlen_n=varlen_n,
        gather_A=gather_A,
    )
    pa_n = cute.sym_int() if gemm_cls_name in ("gated", "grouped_n_contract") else n
    div_pa = div_for_dtype(postact_dtype)
    pa_leading_dim = 1 if gemm_cls_name in ("gated", "grouped_n_contract") else pa_leading
    pa_shape = (m, pa_n) if varlen_m or varlen_n else (m, pa_n, l)
    mAuxOut = fake_tensor(postact_dtype, pa_shape, leading_dim=pa_leading_dim, divisibility=div_pa)

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
    if local_reduce_ndim == 3 and local_reduce_dim == 1:
        mColVecReduce = fake_tensor(
            local_reduce_dtype,
            (l, m, cute.sym_int()),
            leading_dim=2,
            divisibility=1,
        )
        mRowVecReduce = None
    elif local_reduce_ndim == 2 and local_reduce_dim == 1:
        mColVecReduce = fake_tensor(
            local_reduce_dtype,
            (m, cute.sym_int()),
            leading_dim=1,
            divisibility=1,
        )
        mRowVecReduce = None
    elif local_reduce_ndim == 3 and local_reduce_dim == 0:
        mColVecReduce = None
        mRowVecReduce = fake_tensor(
            local_reduce_dtype,
            (l, cute.sym_int(), n),
            leading_dim=2,
            divisibility=1,
        )
    elif local_reduce_ndim == 2 and local_reduce_dim == 0:
        mColVecReduce = None
        mRowVecReduce = fake_tensor(
            local_reduce_dtype,
            (cute.sym_int(), n),
            leading_dim=1,
            divisibility=1,
        )
    else:
        mColVecReduce = None
        mRowVecReduce = None

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
        mAuxOut,
        act_fn,
        tensor_epilogue_fn,
        tensor_epilogue_uses_c,
        tensor_epilogue_returns_aux,
        tensor_epilogue_arg_kinds,
        local_reduce_feeds_main,
        local_reduce_source_from_epilogue,
        local_reduce_group,
        local_reduce_dim,
        local_reduce_op,
        local_reduce_scale,
        local_reduce_max_power,
        alpha=fake_scalar(alpha_mode, Float32),
        beta=fake_scalar(beta_mode, Float32),
        mRowVecBroadcast=mRowVec,
        mColVecBroadcast=mColVec,
        mTensorEpilogueRowVecBroadcasts=mTensorEpilogueRowVecs,
        mTensorEpilogueColVecBroadcasts=mTensorEpilogueColVecs,
        mTensorEpilogueTiles=mTensorEpilogueTiles,
        mColVecReduce=mColVecReduce,
        mRowVecReduce=mRowVecReduce,
        rounding_mode=rounding_mode,
        sr_seed=fake_scalar(sr_seed_mode),
    )
    scheduler_args = make_fake_scheduler_args(
        (is_dynamic_persistent and device_capacity[0] == 9), False, l
    )
    varlen_args = make_fake_varlen_args(
        varlen_m,
        varlen_k,
        gather_A,
        m if varlen_m else (k if varlen_k else None),
        varlen_n=varlen_n,
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
    PostAct: Tensor,  # (l, m, n) or (total_m, n//2) if gated
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
    cu_seqlens_k: Optional[Tensor] = None,  # (l+1,) cumulative sum of k values for variable length
    cu_seqlens_n: Optional[Tensor] = None,  # (l+1,) cumulative sum of n values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) or (total_k,) if gather_A with varlen
    rounding_mode: int = RoundingMode.RN,
    sr_seed: int | Tensor = 0,
    use_tma_gather: bool = False,
    concat_layout: tuple | None = None,
    tensor_epilogue_fn: Optional[Callable] = None,
    tensor_epilogue_key: Optional[str] = None,
    tensor_epilogue_uses_c: bool = False,
    tensor_epilogue_returns_aux: bool = False,
    tensor_epilogue_arg_kinds: tuple[str, ...] = (),
    tensor_epilogue_rowvec_biases: tuple[Tensor, ...] = (),
    tensor_epilogue_colvec_biases: tuple[Tensor, ...] = (),
    tensor_epilogue_tile_biases: tuple[Tensor, ...] = (),
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    local_reduce_out: Optional[Tensor] = None,
    local_reduce_feeds_main: bool = False,
    local_reduce_source_from_epilogue: bool = False,
    local_reduce_group: int = 0,
    local_reduce_dim: int = 1,
    local_reduce_op: str = "sum",
    local_reduce_scale: float = 1.0,
    local_reduce_max_power: int = 8,
    main_output_transform_group: int | None = None,
) -> None:
    if tensor_epilogue_fn is not None:
        assert activation is None, "tensor_epilogue_fn and activation are mutually exclusive"
        if main_output_transform_group is not None:
            if main_output_transform_group not in (2, 4):
                raise NotImplementedError(
                    "grouped_n_contract currently supports only groups 2 and 4"
                )
            gemm_cls_name = "grouped_n_contract"
        else:
            gemm_cls_name = "act"
    elif activation in gate_fn_map:
        gemm_cls_name = "gated"
    else:
        assert activation in act_fn_map, f"Unsupported activation {activation}"
        gemm_cls_name = "act"

    varlen_m = cu_seqlens_m is not None
    varlen_k = cu_seqlens_k is not None
    varlen_n = cu_seqlens_n is not None
    assert sum((varlen_m, varlen_k, varlen_n)) <= 1, "Only one of cu_seqlens_m, cu_seqlens_k, and cu_seqlens_n"
    gather_A = A_idx is not None
    if varlen_m:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    if varlen_k:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"
    if varlen_n:
        ensure_varlen_n_supported(A)
        assert persistent, "varlen_n requires persistent=True"
        assert A.stride(-1) == 1, "varlen_n requires public A to be k-major"
        assert B.stride(-2) == 1, "varlen_n requires B to be n-major after transpose"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_n requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_n requires PostAct to be n-major"
        assert A_idx is None, "gather_A is not supported with varlen_n"
        assert C is None and beta == 1.0, "C/beta are not supported with varlen_n"
        assert rowvec_bias is None and colvec_bias is None, "biases are not supported with varlen_n"
        assert not tensor_epilogue_rowvec_biases and not tensor_epilogue_colvec_biases
        assert not tensor_epilogue_tile_biases, "tile epilogue args are not supported with varlen_n"
        assert local_reduce_out is None, "local reductions are not supported with varlen_n"
        assert main_output_transform_group is None, "shape-changing epilogues are not supported with varlen_n"
    if gather_A:
        assert varlen_m or varlen_k, "gather_A requires varlen"
        assert cluster_N == 1, "gather_A requires cluster_N=1"

    A_p, B_p, D_p, C_p = perm3d(
        A, B, D, C, varlen_m=varlen_m, varlen_k=varlen_k, varlen_n=varlen_n
    )
    PostAct_p = PostAct if varlen_n else perm3d_single(PostAct, varlen_m)
    tensor_epilogue_tile_biases_p = tuple(
        perm3d_single(tensor, varlen_m) for tensor in tensor_epilogue_tile_biases
    )

    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n") if D_p is not None else None
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    postact_major = get_major(PostAct_p, "m", "n")

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    if not set(tensor_epilogue_arg_kinds) <= {"tile", "row", "col"}:
        raise NotImplementedError(
            f"QUACK tensor epilogues support only tile/row/col aux tensors, got {tensor_epilogue_arg_kinds}"
        )
    tensor_epilogue_arg_kind_codes = tuple(
        {"tile": 1, "row": 2, "col": 3}[kind] for kind in tensor_epilogue_arg_kinds
    )
    colvec_ndim = colvec_bias.ndim if colvec_bias is not None else 0
    local_reduce_ndim = local_reduce_out.ndim if local_reduce_out is not None else 0
    local_reduce_op_code = {
        "sum": 0,
        "amax_abs": 1,
        "mx_e8m0_scale": 2,
        "nvfp4_e4m3_scale": 3,
        "copy": 4,
    }[local_reduce_op]

    device_capacity = get_device_capacity(A.device)
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
    concat_layout = tuple(sorted(concat_layout)) if concat_layout else ()
    compiled_fn = _compile_gemm_act(
        a_dtype,
        b_dtype,
        d_dtype,
        c_dtype,
        postact_dtype,
        a_major,
        b_major,
        d_major,
        c_major,
        postact_major,
        (tile_M, tile_N, tile_K) if tile_K is not None else (tile_M, tile_N),
        (cluster_M, cluster_N, 1),
        pingpong,
        persistent,
        is_dynamic_persistent,
        activation,
        tensor_epilogue_fn,
        tensor_epilogue_key if tensor_epilogue_key is not None else repr(tensor_epilogue_fn),
        tensor_epilogue_uses_c,
        tensor_epilogue_returns_aux,
        tensor_epilogue_arg_kind_codes,
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_rowvec_biases),
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_colvec_biases),
        tuple(tensor.ndim for tensor in tensor_epilogue_colvec_biases),
        tuple(torch2cute_dtype_map[tensor.dtype] for tensor in tensor_epilogue_tile_biases_p),
        tuple(get_major(tensor, "m", "n") for tensor in tensor_epilogue_tile_biases_p),
        alpha_mode,
        beta_mode,
        torch2cute_dtype_map[rowvec_bias.dtype] if rowvec_bias is not None else None,
        torch2cute_dtype_map[colvec_bias.dtype] if colvec_bias is not None else None,
        colvec_ndim,
        torch2cute_dtype_map[local_reduce_out.dtype] if local_reduce_out is not None else None,
        local_reduce_ndim,
        local_reduce_feeds_main,
        local_reduce_source_from_epilogue,
        local_reduce_group,
        local_reduce_dim,
        local_reduce_op_code,
        local_reduce_scale,
        local_reduce_max_power,
        0 if main_output_transform_group is None else main_output_transform_group,
        varlen_m,
        varlen_k,
        varlen_n,
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
        PostAct_p,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        alpha=scalar_arg(alpha, alpha_mode, Float32),
        beta=scalar_arg(beta, beta_mode, Float32),
        mRowVecBroadcast=rowvec_bias,
        mColVecBroadcast=colvec_bias,
        mTensorEpilogueRowVecBroadcasts=tensor_epilogue_rowvec_biases or None,
        mTensorEpilogueColVecBroadcasts=tensor_epilogue_colvec_biases or None,
        mTensorEpilogueTiles=tensor_epilogue_tile_biases_p or None,
        mColVecReduce=local_reduce_out if local_reduce_dim == 1 else None,
        mRowVecReduce=local_reduce_out if local_reduce_dim == 0 else None,
        rounding_mode=None,  # Constexpr, pass None at call time
        sr_seed=scalar_arg(sr_seed, sr_seed_mode),
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        max_swizzle_size,
        tile_count_semaphore,
    )
    varlen_args = make_varlen_args(
        cu_seqlens_m, cu_seqlens_k, A_idx, cu_seqlens_n=cu_seqlens_n
    )

    if device_capacity[0] in [10, 11]:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None, None, None)
    else:
        compiled_fn(A_p, B_p, D_p, C_p, epi_args, scheduler_args, varlen_args, None)


gemm_gated = gemm_act
