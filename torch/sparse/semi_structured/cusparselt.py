from __future__ import annotations

from typing import Any, Optional

import torch

from torch.sparse.semi_structured import (
    _SEMI_STRUCTURED_SPARSE_CONFIG,
    SparseSemiStructuredMeta,
    SparseSemiStructuredTensor,
)

__all__ = ["SparseSemiStructuredTensorCUSPARSELT"]


class SparseSemiStructuredTensorCUSPARSELT(  # type: ignore[misc]
    SparseSemiStructuredTensor,
    torch.Tensor,
    metaclass=SparseSemiStructuredMeta
):
    """
    This subclass connects cuSPARSELt to the user.
    """
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(8, 8, 4, 4),
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
    ) -> SparseSemiStructuredTensorCUSPARSELT:
        torch.sparse.SparseSemiStructuredTensor._show_warning()

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
        return ["compressed_tensor_cusparselt"], (
            self.original_shape,
            self.transposed,
            self.fuse_transpose,
            self.alg_id_cusparselt,
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        original_shape, transposed, fuse_transpose, alg_id_cusparselt = meta
        assert (
            len(inner_tensors) == 1
        ), f"Expected 1 inner tensors but got {len(inner_tensors)}"
        compressed_tensor_cusparselt = inner_tensors["compressed_tensor_cusparselt"]

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
            res = torch.ops.aten.addmm.default(
                bias, input_tensor_2d, weight.t(), **kwargs
            )
            return res.view(*shape[:-1], -1)

        if func in {torch.ops.aten.addmm.default, torch.ops.aten.mm.default}:
            if func is torch.ops.aten.addmm.default:
                bias, input_A, input_B = args
            if func is torch.ops.aten.mm.default:
                input_A, input_B = args
                bias = None

            # first element sparse
            if isinstance(input_A, cls) and not input_A.transposed:
                row, col = input_B.shape
                input_B_padded = cls._pad_dense_input(input_B)
                res = torch._cslt_sparse_mm(
                    input_A.compressed_tensor_cusparselt,
                    input_B_padded,
                    bias=bias,  # type: ignore[arg-type]
                    alg_id=input_A.alg_id_cusparselt,
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
                    alg_id=input_B.alg_id_cusparselt,
                )
                res = res if input_B.fuse_transpose else res.t()
                return res[:row, :]

        # handle values
        if func is torch.ops.aten.values.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return (
                args[0].compressed_tensor_cusparselt[:num_kept_elements].view(m, k // 2)
            )

        # handle indices
        if func is torch.ops.aten.indices.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            metadata = (
                args[0].compressed_tensor_cusparselt[num_kept_elements:].view(m, -1)
            )
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
        return torch.mm(self, torch.eye(col, dtype=self.dtype, device=self.device))
