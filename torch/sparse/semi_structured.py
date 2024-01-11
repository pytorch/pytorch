import warnings
from collections import namedtuple
from typing import Any, Optional

import torch

__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG", "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols"
)

class SparseSemiStructuredTensor(torch.Tensor):
    """This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    Currently, this class supports 2:4 sparsity for int8, float16 and bfloat16 dtypes.
    We also support 1:2 sparsity for float32 dtype.

    This subclass stores the dense tensor in a compressed form by only storing the specified elements and corresponding metadata.

    The subclass supports two backend, either CUTLASS or cuSPASRELt.

    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:

    compressed tensor = [ specified elements of original tensor | metadata ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    For CUTLASS backend, elements of original tensor and metadata are kept in separate tensors.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_linear
    and sparse_semi_structured_from_dense for conversion to the compressed format.

    When PyTorch is compiled with cuSPARSELt support, this subclass will call into _cslt_sparse_mm for sparse mm and
    _cslt_compress to convert into the compressed format.
    """

    _FUSE_TRANSPOSE = False
    _FORCE_CUTLASS = True
    _PROTOTYPE_WARNING_SHOWN = False

    @staticmethod
    def __new__(
        cls,
        original_shape: Optional[torch.Size],
        transposed: bool = False,
        sparse_tensor_cutlass: Optional[torch.Tensor] = None,
        meta_tensor_cutlass: Optional[torch.Tensor] = None,
        compressed_tensor_cusparselt: Optional[torch.Tensor] = None,
        fuse_transpose_cusparselt: bool = False,
        alg_id_cusparselt: int = 0,
    ):
        """
        Create a new instance of the class.

        When original_tensor is passed in, we compress it and store the compresed representation.
        We can also create new instance of the class from the compressed representation without the original tensor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            compressed_tensor_cusparselt: For cuSPARSELt backend, a flattened tensor to store the specified elements and metadata.
            sparse_tensor_cutlass: For CUTLASS backend, tensor to store the speficied elements.
            meta_tensor_cutlass: For CUTLASS backend, tensor to store metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.

        """
        assert compressed_tensor_cusparselt is None or (sparse_tensor_cutlass is None and meta_tensor_cutlass is None)

        if not cls._PROTOTYPE_WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
            )
            cls._PROTOTYPE_WARNING_SHOWN = True

        if compressed_tensor_cusparselt is not None:
            previous_tensor = compressed_tensor_cusparselt
        elif sparse_tensor_cutlass is not None:
            previous_tensor = sparse_tensor_cutlass
        else:
            raise ValueError("All of the tensor arguments are None!")

        kwargs = {}
        kwargs["device"] = previous_tensor.device  # type: ignore[assignment]
        kwargs["dtype"] = previous_tensor.dtype  # type: ignore[assignment]
        kwargs["layout"] = previous_tensor.layout  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]

        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_shape: Optional[torch.Size],
        transposed: bool = False,
        sparse_tensor_cutlass: Optional[torch.Tensor] = None,
        meta_tensor_cutlass: Optional[torch.Tensor] = None,
        compressed_tensor_cusparselt: Optional[torch.Tensor] = None,
        fuse_transpose_cusparselt: bool = False,
        alg_id_cusparselt: int = 0,
    ) -> None:
        self.compressed_tensor_cusparselt = compressed_tensor_cusparselt
        self.sparse_tensor_cutlass = sparse_tensor_cutlass
        self.meta_tensor_cutlass = meta_tensor_cutlass
        self.original_shape = original_shape
        self.transposed = transposed
        self.fuse_transpose_cusparselt = fuse_transpose_cusparselt
        self.alg_id_cusparselt = alg_id_cusparselt

    @classmethod
    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # only 2d matmul
        assert dense_input.dim() == 2

        # check shape
        m, n = dense_input.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_cols

        # calculate padding
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    def __repr__(self) -> str:  # type: ignore[override]
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        assert hasattr(self, "shape")
        assert hasattr(self, "transposed")
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"transposed={self.transposed})"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        """Overload __torch_dispatch__ to use torch._sparse_semi_structured_linear.

        `torch.structured_sparse_linear` uses accelerated sparse CUTLASS kernels.
        In the future we plan to also add in support for cuSPARSELt kernels.

        Args:
            func: The function being dispatched.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation.

        Raises:
            NotImplementedError: If the dispatched operation is not implemented.
        """
        if func is torch.ops.aten.values.default:
            return args[0].values()

        if func is torch.ops.aten.indices.default:
            return args[0].indices()

        # Since this code runs below autograd, a detach corresponds to only returning a new object
        if func is torch.ops.aten.detach.default:
            return args[0].__class__(
                args[0].shape,
                transposed=args[0].transposed,
                sparse_tensor_cutlass=args[0].sparse_tensor_cutlass,
                meta_tensor_cutlass=args[0].meta_tensor_cutlass,
                compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt,
                fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
                alg_id_cusparselt=args[0].alg_id_cusparselt,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
        # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
        if func is torch.ops.aten.t.default:
            return args[0].__class__(
                torch.Size([args[0].shape[-1], args[0].shape[0]]),
                transposed=not args[0].transposed,
                sparse_tensor_cutlass=args[0].sparse_tensor_cutlass,
                meta_tensor_cutlass=args[0].meta_tensor_cutlass,
                compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt,
                fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
                alg_id_cusparselt=args[0].alg_id_cusparselt,
            )

        # When torch is run with inference mode, pytorch does not decompose torch.ops.aten.linear into a .t() and addmm(),
        # so we must match the aten.linear op. In this case, we need to explicitly handle collapsing to 2d.
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

            if isinstance(input_A, cls) and not input_A.transposed:
                row, col = input_B.shape
                input_B_padded = cls._pad_dense_input(input_B)
                res = input_A.sparse_addmm(input_B_padded, bias)
                return res[:, :col]

            elif isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = cls._pad_dense_input(input_A)
                res = input_B.sparse_addmm(input_A_padded.t(), bias)
                res = res if input_B.fuse_transpose_cusparselt else res.t():
                return res[:row, :]

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )

        raise NotImplementedError(error_string)

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    """This class provides the CUTLASS implementation of semi-structured (2:4) sparsity for acceleration on GPUs.
    It connects the user to `_sparse_semi_structured_linear`, which uses CUTLASS for accelerated sparse matmul.

    For CUTLASS the compressed representation is stored separately, as two distinct tensors:
    - sparse_tensor_cutlass (holds the specified elements of original tensor)
    - meta_tensor_cutlass (holds the metadata bitmask)
    """

    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4),
    }

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[torch.Size, bool]]:
        return ["sparse_tensor_cutlass", "meta_tensor_cutlass"], (
            self.original_shape,
            self.transposed,
        )

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors, meta, outer_size, outer_stride
    ) -> SparseSemiStructuredTensor:
        original_shape, transposed = meta

        assert (
            len(inner_tensors) == 2
        ), f"Expected 2 inner tensors but got {len(inner_tensors)}"
        sparse_tensor_cutlass = inner_tensors["sparse_tensor_cutlass"]
        meta_tensor_cutlass = inner_tensors["meta_tensor_cutlass"]

        return SparseSemiStructuredTensorCUTLASS(
            original_shape,
            sparse_tensor_cutlass=sparse_tensor_cutlass,
            meta_tensor_cutlass=meta_tensor_cutlass,
            transposed=transposed,
        )

    @classmethod
    def from_dense(cls, original_tensor):
        cls._validate_device_dim_dtype_shape(original_tensor)
        sparse_tensor_cutlass, meta_tensor_cutlass = sparse_semi_structured_from_dense_cutlass(original_tensor)
        return cls(original_tensor.shape,
                   sparse_tensor_cutlass=sparse_tensor_cutlass,
                   meta_tensor_cutlass=meta_tensor_cutlass)

    def to_dense(self):
        return sparse_semi_structured_to_dense_cutlass(
            self.sparse_tensor_cutlass,
            self.meta_tensor_cutlass,
        )

    def sparse_addmm(self, dense, bias):
        return torch._sparse_semi_structured_linear(
            dense.t(),
            self.sparse_tensor_cutlass,
            self.meta_tensor_cutlass,
            bias=bias).t()

    def values(self):
        return self.sparse_tensor_cutlass.detach()

    def indices(self):
        return self.meta_tensor_cutlass

class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
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
            original_shape,
            compressed_tensor_cusparselt=compressed_tensor_cusparselt,
            transposed=transposed,
            fuse_transpose=fuse_transpose,
            alg_id_cusparselt=alg_id_cusparselt,
        )

    @classmethod
    def from_dense(cls, original_tensor):
        cls._validate_device_dim_dtype_shape(original_tensor)
        compressed_tensor_cusparselt = torch._cslt_compress(original_tensor)
        return cls(compressed_tensor_cusparselt=compressed_tensor_cusparselt, original_tensor.shape)

    def to_dense(self):
        col = self.shape[-1]
        return torch.mm(self, torch.eye(col, dtype=self.dtype, device=self.device))

    def sparse_addmm(self, dense, bias):
        return torch._cslt_sparse_mm(
            dense,
            self.compressed_tensor_cusparselt,
            bias=bias
            fuse_transpose=self.fuse_transpose_cusparselt,
            alg_id=self.alg_id_cusparselt)

    def values(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        return self.compressed_tensor_cusparselt[:num_kept_elements:].view(m, -1)

    def indices(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        metadata = self.compressed_tensor_cusparselt[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if self.dtype == torch.int32 else torch.int16)

def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
) -> Any:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return either
        1. a SparseSemiStructuredTensor if the input tensor is already in the correct format
        2. a regular SparseTensor if the input tensor was not in the correct format

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
    sparse_subclass = SparseSemiStructuredTensorCUTLASS if SparseSemiStructuredTensor._FORCE_CUTLASS else SparseSemiStructuredTensorCUSPARSELT
    return sparse_subclass.from_dense(original_tensor)
