import warnings
from collections import namedtuple
from typing import Any, Optional, Union, TypeVar, Type
from torch.sparse.utils import SparseSemiStructuredMeta, SparseSemiStructuredTensor

import torch

__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS"
    "SparseSemiStructuredTensorCUSPARSELT"
    "to_sparse_semi_structured",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG", "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols"
)

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor, torch.Tensor, metaclass=SparseSemiStructuredMeta):

    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4)
    }

    @staticmethod
    def __new__(
        cls,
        sparse_tensor_cutlass: torch.Tensor,
        meta_tensor_cutlass: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
        transposed: bool = False,
    ):
        SparseSemiStructuredTensor._show_warning()

        kwargs = {
            "device": sparse_tensor_cutlass.device,  # type: ignore[assignment]
            "dtype": sparse_tensor_cutlass.dtype,  # type: ignore[assignment]
            "layout": sparse_tensor_cutlass.layout,  # type: ignore[assignment]
            "requires_grad": False,  # type: ignore[assignment]
        }
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        sparse_tensor_cutlass: torch.Tensor,
        meta_tensor_cutlass: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
        transposed: bool = False,
    ) -> None:
        self.sparse_tensor_cutlass = sparse_tensor_cutlass
        self.meta_tensor_cutlass = meta_tensor_cutlass
        self.original_shape = original_shape
        self.transposed = transposed

    def __tensor_flatten__(self):
        return ['sparse_tensor_cutlass', 'meta_tensor_cutlass'], (self.original_shape, self.transposed)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        original_shape, transposed = meta

        assert len(inner_tensors) == 2, f"Expected 2 inner tensors but got {len(inner_tensors)}"
        sparse_tensor_cutlass = inner_tensors['sparse_tensor_cutlass']
        meta_tensor_cutlass = inner_tensors['meta_tensor_cutlass']

        return SparseSemiStructuredTensorCUTLASS(
            sparse_tensor_cutlass,
            meta_tensor_cutlass,
            original_shape=original_shape,
            transposed=transposed,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        # Since this code runs below autograd, a detach corresponds to only returning a new object
        if func is torch.ops.aten.detach.default:
            return cls(
                args[0].sparse_tensor_cutlass,
                args[0].meta_tensor_cutlass,
                original_shape=args[0].shape,
                transposed=args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
        # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
        if func is torch.ops.aten.t.default:
            return cls(
                args[0].sparse_tensor_cutlass,
                args[0].meta_tensor_cutlass,
                original_shape=torch.Size([args[0].shape[1], args[0].shape[0]]),
                transposed=not args[0].transposed,
            )

        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            shape = input_tensor.shape
            input_tensor_2d = input_tensor.view(-1, shape[-1])
            res = torch.addmm(bias, input_tensor_2d, weight.t(), **kwargs)
            return res.view(*shape[:-1], -1)

        if func in {torch.ops.aten.addmm.default, torch.ops.aten.mm.default}:
            if func is torch.ops.aten.addmm.default:
                bias, input_A, input_B = args
            if func is torch.ops.aten.mm.default:
                bias, (input_A, input_B) = None, args

            # first element sparse
            if isinstance(input_A, cls) and not input_A.transposed:
                row, col = input_B.shape
                input_B_padded = cls._pad_dense_input(input_B)
                res = torch._sparse_semi_structured_linear(
                    input_B_padded.t(),
                    input_A.sparse_tensor_cutlass,
                    input_A.meta_tensor_cutlass,
                    bias=bias
                ).t()
                return res[:, :col]

            # second element sparse
            elif isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = cls._pad_dense_input(input_A)
                res = torch._sparse_semi_structured_linear(
                    input_A_padded,
                    input_B.sparse_tensor_cutlass,
                    input_B.meta_tensor_cutlass,
                    bias=bias
                )
                return res[:row, :]

        # handle values
        if func is torch.ops.aten.values.default:
            return args[0].sparse_tensor_cutlass.detach()

        # handle indices
        if func is torch.ops.aten.indices.default:
            return args[0].meta_tensor_cutlass

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )
        raise NotImplementedError(error_string)

    @classmethod
    def from_dense(cls, original_tensor):
        cls._validate_device_dim_dtype_shape(original_tensor)
        from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
        sparse_tensor_cutlass, meta_tensor_cutlass = sparse_semi_structured_from_dense_cutlass(original_tensor)
        return cls(sparse_tensor_cutlass, meta_tensor_cutlass, original_shape=original_tensor.shape)

    def to_dense(self):
        if self.sparse_tensor_cutlass.dtype == torch.float32:
            raise RuntimeError("Converting to dense for torch.float32 datatype is not yet supported by CUTLASS backend!")

        from torch.sparse._semi_structured_conversions import (
            sparse_semi_structured_to_dense_cutlass,
        )

        return sparse_semi_structured_to_dense_cutlass(
            self.sparse_tensor_cutlass,
            self.meta_tensor_cutlass,
        )


class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor, torch.Tensor, metaclass=SparseSemiStructuredMeta):

    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(8, 8, 4, 4)
    }
    _FUSE_TRANSPOSE = False
    _DEFAULT_ALG_ID = 0

    @staticmethod
    def __new__(
        cls,
        compressed_tensor_cusparselt: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
        transposed: bool = False,
        fuse_transpose: bool = False,
        alg_id_cusparselt: int = 0,
    ):
        SparseSemiStructuredTensor._show_warning()

        kwargs = {}
        kwargs["device"] = compressed_tensor_cusparselt.device  # type: ignore[assignment]
        kwargs["dtype"] = compressed_tensor_cusparselt.dtype  # type: ignore[assignment]
        kwargs["layout"] = compressed_tensor_cusparselt.layout  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]

        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        compressed_tensor_cusparselt: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
        transposed: bool = False,
        fuse_transpose: bool = False,
        alg_id_cusparselt: int = 0,
    ) -> None:
        self.compressed_tensor_cusparselt = compressed_tensor_cusparselt
        self.original_shape = original_shape
        self.transposed = transposed
        self.fuse_transpose = fuse_transpose
        self.alg_id_cusparselt = alg_id_cusparselt

    def __tensor_flatten__(self):
        return ['compressed_tensor_cusparselt'], (self.original_shape, self.transposed, self.fuse_transpose, self.alg_id_cusparselt)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        original_shape, transposed, fuse_transpose, alg_id_cusparselt = meta
        assert len(inner_tensors) == 1, f"Expected 1 inner tensors but got {len(inner_tensors)}"
        compressed_tensor_cusparselt = inner_tensors['compressed_tensor_cusparselt']

        return SparseSemiStructuredTensorCUSPARSELT(
            compressed_tensor_cusparselt,
            original_shape=original_shape,
            transposed=transposed,
            fuse_transpose=fuse_transpose,
            alg_id_cusparselt=alg_id_cusparselt,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        # Since this code runs below autograd, a detach corresponds to only returning a new object
        if func is torch.ops.aten.detach.default:
            return cls(
                args[0].compressed_tensor_cusparselt,
                original_shape=args[0].shape,
                transposed=args[0].transposed,
                fuse_transpose=args[0].fuse_transpose,
                alg_id_cusparselt=args[0].alg_id_cusparselt,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
        # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
        if func is torch.ops.aten.t.default:
            return cls(
                args[0].compressed_tensor_cusparselt,
                # transpose shape
                original_shape=torch.Size([args[0].shape[1], args[0].shape[0]]),
                transposed=not args[0].transposed,
                fuse_transpose=args[0].fuse_transpose,
                alg_id_cusparselt=args[0].alg_id_cusparselt,
            )

        # handle linear case specially
        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            shape = input_tensor.shape
            input_tensor_2d = input_tensor.view(-1, shape[-1])
            res = torch.ops.aten.addmm.default(bias, input_tensor_2d, weight.t(), **kwargs)
            return res.view(*shape[:-1], -1)

        if func in {torch.ops.aten.addmm.default, torch.ops.aten.mm.default}:
            if func is torch.ops.aten.addmm.default:
                bias, input_A, input_B = args
            if func is torch.ops.aten.mm.default:
                input_A, input_B = args
                bias=None

            # first element sparse
            if isinstance(input_A, cls) and not input_A.transposed:
                row, col = input_B.shape
                input_B_padded = cls._pad_dense_input(input_B)
                res = torch._cslt_sparse_mm(
                    input_A.compressed_tensor_cusparselt,
                    input_B_padded,
                    bias=bias,  # type: ignore[arg-type]
                    alg_id=input_A.alg_id_cusparselt
                )
                return res[:, :col]

            # second element sparse
            elif isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = cls._pad_dense_input(input_A)
                res = torch._cslt_sparse_mm(
                    input_B.compressed_tensor_cusparselt,
                    input_A_padded.t(),
                    bias=bias,  # type: ignore[arg-type]
                    transpose_result=input_B.fuse_transpose,
                    alg_id=input_B.alg_id_cusparselt
                )
                res = res if input_B.fuse_transpose else res.t()
                return res[:row, :]

        # handle values
        if func is torch.ops.aten.values.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return args[0].compressed_tensor_cusparselt[:num_kept_elements].view(m, k // 2)

        # handle indices
        if func is torch.ops.aten.indices.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            metadata = args[0].compressed_tensor_cusparselt[num_kept_elements:].view(m, -1)
            metadata_dtype = torch.int32 if args[0].dtype == torch.int8 else torch.int16
            return metadata.view(metadata_dtype)

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )
        raise NotImplementedError(error_string)

    @classmethod
    def from_dense(cls, original_tensor):
        cls._validate_device_dim_dtype_shape(original_tensor)
        compressed_tensor_cusparselt = torch._cslt_compress(original_tensor)
        return cls(compressed_tensor_cusparselt, original_shape=original_tensor.shape)

    def to_dense(self):
        raise RuntimeError("Converting to dense is not yet supported by cuSPARSELt backend!")


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
) -> Any:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of a block size given the dtype

    - torch.float16  (r, c) must be >= and a multiple of 64
    - torch.int8     (r, c) must be >= and a multiple of 128

    Args:
        original_tensor (Tensor): the dense tensor to convert

    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor

    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
            metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
       dtype=torch.int16))
    """
    if SparseSemiStructuredTensor._FORCE_CUTLASS:
        return SparseSemiStructuredTensorCUTLASS.from_dense(original_tensor)
    else:
        return SparseSemiStructuredTensorCUSPARSELT.from_dense(original_tensor)
