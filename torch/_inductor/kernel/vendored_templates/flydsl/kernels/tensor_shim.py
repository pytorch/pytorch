# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import torch

import flydsl.compiler as flyc
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import arith, buffer_ops, range_constexpr, vector
from flydsl.expr.typing import T


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


def get_dtype_str(dtype):
    if dtype == torch.float:
        return "f32"
    elif dtype == torch.half:
        return "f16"
    elif dtype == torch.bfloat16:
        return "bf16"


def get_dtype_in_kernel(dtype: str):
    if dtype == "f32":
        return T.f32
    elif dtype == "f16":
        return T.f16
    elif dtype == "bf16":
        return T.bf16


def get_dtype_vec_size(dtype: str):
    if dtype == "f32":
        return 4
    elif dtype == "f16":
        return 8
    elif dtype == "bf16":
        return 8


def get_dtype_bytes(dtype: str):
    if dtype == "f32":
        return 4
    elif dtype == "f16":
        return 2
    elif dtype == "bf16":
        return 2


class TensorView:
    def __init__(self, dtype, shape, stride, base_offset, load_impl, store_impl):
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple(
                (
                    np.cumprod(shape[::-1])[::-1].tolist()
                    + [
                        1,
                    ]
                )[1:]
            )
        else:
            self.stride = stride
        self.base_offset = base_offset
        self.load_impl = load_impl
        self.store_impl = store_impl

    def _linear_offset(self, idxs):
        slice_shape = []
        slice_stride = []
        d_offset = self.base_offset
        for i in range_constexpr(len(idxs)):
            md_id = idxs[i]
            if md_id is None:
                slice_shape.append(self.shape[i])
                slice_stride.append(self.stride[i])
            elif isinstance(md_id, int):
                d_offset = d_offset + md_id * self.stride[i]
            else:
                d_offset = d_offset + md_id * self.stride[i]
        if len(slice_shape) > 0:
            return d_offset, tuple(slice_shape), tuple(slice_stride)
        else:
            return (d_offset,)

    def _lazy_init(self):
        pass

    def __repr__(self):
        return f"TensorView(offset={self.base_offset}, shape={self.shape}, stride={self.stride}, dtype={self.dtype})"

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        if len(offset) == 1:
            return self.load_impl(offset[0])
        else:
            return TensorView(
                self.dtype,
                offset[1],
                offset[2],
                offset[0],
                self.load_impl,
                self.store_impl,
            )

    def __setitem__(self, idxs, value):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value)

    def vec_load(self, idxs, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return self.load_impl(offset[0], vec_size=vec_size)

    def vec_store(self, idxs, value, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value, vec_size=vec_size)

    def linear_offset(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return offset[0]

    def local_tile(self, tile_shape, tile_idxs):
        d_offset = self.base_offset
        stride = []
        for i in range_constexpr(len(tile_idxs)):
            d_offset = d_offset + tile_idxs[i] * tile_shape[i] * self.stride[i]
            stride.append(self.stride[i])
        return TensorView(
            self.dtype,
            tile_shape,
            tuple(stride),
            d_offset,
            self.load_impl,
            self.store_impl,
        )

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        src_tensor._lazy_init()
        ndim = len(thread_layout)
        src_offset = src_tensor.base_offset
        dst_offset = self.base_offset
        for d in range_constexpr(ndim):
            src_offset = src_offset + thread_idxs[d] * value_layout[d] * src_tensor.stride[d]
            dst_offset = dst_offset + thread_idxs[d] * value_layout[d] * self.stride[d]
        value_layout_v = value_layout[:-1] + (value_layout[-1] // vec_size,)
        coords = tuple(product(*(range_constexpr(s) for s in value_layout_v)))
        for coord in coords:
            src_vec_offset = src_offset
            dst_vec_offset = dst_offset
            for d in range_constexpr(len(coord)):
                if d == len(coord) - 1:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d] * vec_size
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d] * vec_size
                else:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d]
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d]
            value = src_tensor.load_impl(src_vec_offset, vec_size=vec_size)
            self.store_impl(dst_vec_offset, value, vec_size=vec_size)


class TensorBase(ABC):
    def __init__(self, dtype, shape, stride=None, base_offset=0):
        self.tensor_view = None
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.base_offset = base_offset

    @abstractmethod
    def load(self, offset):
        return None

    @abstractmethod
    def store(self, offset, value):
        pass

    def _lazy_init(self):
        if self.tensor_view is None:
            self.tensor_view = TensorView(
                self.dtype,
                self.shape,
                self.stride,
                self.base_offset,
                self.load,
                self.store,
            )
            self.stride = self.tensor_view.stride
            self.load_impl = self.tensor_view.load_impl
            self.store_impl = self.tensor_view.store_impl

    def __repr__(self):
        self._lazy_init()
        return self.tensor_view.__repr__()

    def __getitem__(self, idxs):
        self._lazy_init()
        return self.tensor_view[idxs]

    def __setitem__(self, idxs, value):
        self._lazy_init()
        self.tensor_view[idxs] = value

    def vec_load(self, idxs, vec_size):
        self._lazy_init()
        return self.tensor_view.vec_load(idxs, vec_size)

    def vec_store(self, idxs, value, vec_size):
        self._lazy_init()
        self.tensor_view.vec_store(idxs, value, vec_size)

    def linear_offset(self, idxs):
        self._lazy_init()
        return self.tensor_view.linear_offset(idxs)

    def local_tile(self, tile_shape, tile_idxs):
        self._lazy_init()
        return self.tensor_view.local_tile(tile_shape, tile_idxs)

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        self._lazy_init()
        self.tensor_view.copy_(src_tensor, thread_layout, value_layout, thread_idxs, vec_size)


class TorchTensor(TensorBase):
    def __init__(self, torch_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.torch_tensor = torch_tensor

    def load(self, offset, vec_size=1):
        return self.torch_tensor.view(-1)[offset : offset + vec_size]

    def store(self, offset, value, vec_size=1):
        self.torch_tensor.view(-1)[offset : offset + vec_size] = value


class GTensor(TensorBase):
    def __init__(
        self,
        memref,
        dtype,
        shape,
        stride=None,
        base_offset=0,
        cache_modifier=0,
        static_bytes_offset_i64=None,
    ):
        super().__init__(dtype, shape, stride, base_offset)
        if static_bytes_offset_i64 is None:
            self.rsrc = buffer_ops.create_buffer_resource(memref, max_size=True)
        else:
            array_base_i64 = self.get_llvm_ptr(memref, (static_bytes_offset_i64))
            self.rsrc = buffer_ops.create_buffer_resource_from_addr(array_base_i64)
        self.cache_modifier = cache_modifier

    def load(self, offset, vec_size=1):
        return buffer_ops.buffer_load(self.rsrc, offset, vec_width=vec_size, dtype=self.dtype)

    def store(self, offset, value, vec_size=1):
        buffer_ops.buffer_store(value, self.rsrc, offset, cache_modifier=self.cache_modifier)

    def get_llvm_ptr(self, ptr, bytes_offset_i64, ptr_type="!llvm.ptr<1>"):
        bytes_offset_i64 = arith.index_cast(T.i64, bytes_offset_i64)
        _ptr_type = ir.Type.parse(ptr_type)
        base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, extract_to_ir_values(ptr)[0])
        base_ptr = llvm.PtrToIntOp(T.i64, base_ptr).result
        llvm_ptr = llvm.AddOp(base_ptr, bytes_offset_i64, llvm.IntegerOverflowFlags(0)).result
        return llvm_ptr


class STensor(TensorBase):
    def __init__(self, memptr, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.memptr = memptr.get()

    def load(self, offset, vec_size=1):
        vec_t = T.vec(vec_size, self.dtype)
        x = vector.load_op(vec_t, self.memptr, [offset])
        if vec_size > 1:
            return x
        else:
            x = vector.extract(x, static_position=[0], dynamic_position=[])
            return x

    def store(self, offset, value, vec_size=1):
        if vec_size > 1:
            vector.store(value, self.memptr, [offset], alignment=16)
        else:
            vec_t = T.vec(1, self.dtype)
            vec = vector.from_elements(vec_t, [value])
            vector.store(vec, self.memptr, [offset], alignment=16)
