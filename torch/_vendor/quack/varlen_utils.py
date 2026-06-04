# Copyright (c) 2025, Tri Dao.

from typing import Optional, NamedTuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr

from . import copy_utils
from .cute_dsl_utils import mlir_namedtuple


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class VarlenArguments(NamedTuple):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None


@mlir_namedtuple
class VarlenNArguments(NamedTuple):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None
    mCuSeqlensN: Optional[cute.Tensor] = None


def cu_seqlens_n_arg(args) -> Optional[cute.Tensor]:
    return args.mCuSeqlensN if isinstance(args, VarlenNArguments) else None


class VarlenManager:
    @dataclass
    class Params:
        cu_seqlens_m: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        cu_seqlens_n: Optional[cute.Tensor] = None
        mAIdx: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(args, *, loc=None, ip=None) -> "VarlenManager.Params":
            return VarlenManager.Params(
                cu_seqlens_m=args.mCuSeqlensM,
                cu_seqlens_k=args.mCuSeqlensK,
                cu_seqlens_n=cu_seqlens_n_arg(args),
                mAIdx=args.mAIdx,
            )

    def __init__(
        self,
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        len_n_static: Int32,
        last_batch_idx: Int32 = Int32(-1),
        is_group_changed: Boolean = Boolean(True),
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._len_n_static = len_n_static
        self._last_batch_idx = last_batch_idx
        self._is_group_changed = is_group_changed
        self.varlen_m = const_expr(params.cu_seqlens_m is not None)
        self.varlen_k = const_expr(params.cu_seqlens_k is not None)
        self.varlen_n = const_expr(params.cu_seqlens_n is not None)
        self.gather_A = const_expr(params.mAIdx is not None)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args, *, loc=None, ip=None) -> Params:
        assert sum(
            x is not None for x in (args.mCuSeqlensM, args.mCuSeqlensK, cu_seqlens_n_arg(args))
        ) <= 1, "Only support one of varlen_m, varlen_k, or varlen_n"
        return VarlenManager.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        len_m_static: Int32,
        len_k_static: Int32,
        len_n_static: Int32,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenManager":
        return VarlenManager(
            params,
            len_m_static=len_m_static,
            len_k_static=len_k_static,
            len_n_static=len_n_static,
        )

    def len_m(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_m):
            return self.params.cu_seqlens_m[batch_idx + 1] - self.params.cu_seqlens_m[batch_idx]
        else:
            return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_k):
            return self.params.cu_seqlens_k[batch_idx + 1] - self.params.cu_seqlens_k[batch_idx]
        else:
            return self._len_k_static

    def len_n(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_n):
            return self.params.cu_seqlens_n[batch_idx + 1] - self.params.cu_seqlens_n[batch_idx]
        else:
            return self._len_n_static

    def offset_batch_A(self, mA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mA_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx], None), mA_mkl)
        elif const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mA_mkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mA_mk = cute.domain_offset((None, offset), mA_mkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                # rank 3 = 1-extra-dim (ptr_shift), rank 4 = 2-extra-dim
                ptr_shift = const_expr(ragged_rank == 3)
                mA_mk = copy_utils.offset_ragged_tensor(
                    mA_mkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mA_mk = mA_mkl[None, None, batch_idx]
        return mA_mk

    def offset_batch_AIdx(self, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_k):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_k[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_n):
            mAIdx_mk = params.mAIdx[None, batch_idx]
        else:
            mAIdx_mk = params.mAIdx[None, batch_idx]
        return mAIdx_mk

    def offset_batch_SFA(self, mSFA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        """Offset SFA by padded per-expert offset (dQaccum-style).

        The padded offset, in tile units (128 source-M or source-K per tile),
        is simply `cu_seqlens[b] // 128 + b`. (Algebraically identical to
        `(cu_seqlens[b] + b*128) // 128 * 128` / 128.) We pass it as a
        compound coord `(0, offset_tile)` to `domain_offset` so the outer
        rm/rk mode is shifted in tile units — no `* 128` needed, and the
        compiler sees the tile alignment natively.
        """
        params = self.params
        tile = 128
        if const_expr(self.varlen_m):
            offset_tile = params.cu_seqlens_m[batch_idx] // tile + batch_idx
            return cute.domain_offset(((0, offset_tile), None), mSFA_mkl)
        elif const_expr(self.varlen_k):
            offset_tile = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, (0, offset_tile)), mSFA_mkl)
        else:
            return mSFA_mkl[None, None, batch_idx]

    def offset_batch_SFB(self, mSFB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        """Offset SFB by padded per-expert K offset (varlen_k only)."""
        params = self.params
        tile = 128
        if const_expr(self.varlen_k):
            offset_tile = params.cu_seqlens_k[batch_idx] // tile + batch_idx
            return cute.domain_offset((None, (0, offset_tile)), mSFB_nkl)
        else:
            return mSFB_nkl[None, None, batch_idx]

    def offset_batch_B(self, mB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_k):
            offset = params.cu_seqlens_k[batch_idx]
            ragged_rank = const_expr(cute.rank(mB_nkl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mB_nk = cute.domain_offset((None, offset), mB_nkl)
            else:
                length = params.cu_seqlens_k[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mB_nk = copy_utils.offset_ragged_tensor(
                    mB_nkl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        elif const_expr(self.varlen_n):
            offset = params.cu_seqlens_n[batch_idx]
            length = params.cu_seqlens_n[batch_idx + 1] - offset
            ragged_rank = const_expr(cute.rank(mB_nkl))
            if const_expr(ragged_rank == 2):
                mB_offset = cute.domain_offset((offset, None), mB_nkl)
                mB_nk = cute.make_tensor(
                    mB_offset.iterator,
                    cute.make_layout((length, cute.size(mB_nkl, mode=[1])), stride=mB_nkl.stride),
                )
            else:
                ptr_shift = const_expr(ragged_rank == 3)
                mB_nk = copy_utils.offset_ragged_tensor(
                    mB_nkl,
                    offset,
                    length,
                    ragged_dim=0,
                    ptr_shift=ptr_shift,
                )
        else:
            mB_nk = mB_nkl[None, None, batch_idx]
        return mB_nk

    def offset_batch_epi(self, mD_mnl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            offset = params.cu_seqlens_m[batch_idx]
            ragged_rank = const_expr(cute.rank(mD_mnl))
            if const_expr(ragged_rank == 2):  # Didn't create ragged tensor
                mD_mn = cute.domain_offset((offset, None), mD_mnl)
            else:
                length = params.cu_seqlens_m[batch_idx + 1] - offset
                ptr_shift = const_expr(ragged_rank == 3)
                mD_mn = copy_utils.offset_ragged_tensor(
                    mD_mnl,
                    offset,
                    length,
                    ragged_dim=0,
                    ptr_shift=ptr_shift,
                )
        elif const_expr(self.varlen_n):
            offset = params.cu_seqlens_n[batch_idx]
            length = params.cu_seqlens_n[batch_idx + 1] - offset
            ragged_rank = const_expr(cute.rank(mD_mnl))
            if const_expr(ragged_rank == 2):
                mD_offset = cute.domain_offset((None, offset), mD_mnl)
                mD_mn = cute.make_tensor(
                    mD_offset.iterator,
                    cute.make_layout((cute.size(mD_mnl, mode=[0]), length), stride=mD_mnl.stride),
                )
            else:
                ptr_shift = const_expr(ragged_rank == 3)
                mD_mn = copy_utils.offset_ragged_tensor(
                    mD_mnl,
                    offset,
                    length,
                    ragged_dim=1,
                    ptr_shift=ptr_shift,
                )
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        return mD_mn

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self._len_m_static,
            self._len_k_static,
            self._len_n_static,
            self._last_batch_idx,
            self._is_group_changed,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self._len_m_static,
                self._len_k_static,
                self._len_n_static,
                self._last_batch_idx,
                self._is_group_changed,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
