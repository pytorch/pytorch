from collections import namedtuple
import warnings

import torch


__all__ = [
    "to_sparse_semi_structured",
    "SparseSemiStructuredTensor",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple("_SEMI_STRUCTURED_SPARSE_CONFIG", "compression_factor n min_size")
_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG = {
    torch.float16 : _SEMI_STRUCTURED_SPARSE_CONFIG(9, 2, 64),
    torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(10, 2, 128),
}

class SparseSemiStructuredTensor(torch.Tensor):
    """
    This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    For torch.float32, this tensor subclass implements 1:2 semi-structured sparsity and 2:4 semi-structured
    sparsity for torch.int8, torch.float16, and torch.bfloat16 datatypes.

    This subclass stores the dense tensor in a compressed form by only storing the specified elemenets and a metadata mask.
    These two are stored next to each other in one contiguous tensor.

    compressed tensor = [ specified elements of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    The reason why we store in a single compressed tensor vs. a dense tensor is for future compatibilty with cuSPARSELt.

    This subclass also overrides __torch_dispatch__ to use cuSPARSELt for faster matrix multiplications.
    """

    @staticmethod
    def __new__(cls, custom_shape, compressed_tensor, transposed):
        kwargs = {}
        kwargs["device"] = compressed_tensor.device
        kwargs["dtype"] = compressed_tensor.dtype
        kwargs["layout"] = compressed_tensor.layout
        kwargs["requires_grad"] = False

        warnings.warn(
            (
                "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                "and will change in the near future. Please open a Github issue "
                "for features requests and see our documentation on the torch.sparse "
                "module for further information about the project."
            ),
            UserWarning,
        )

        return torch.Tensor._make_wrapper_subclass(cls, custom_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        custom_shape: torch.Size,
        compressed_tensor: torch.Tensor,
        transposed: bool,
    ) -> None:
        """This function is the constuctor for SparseSemiStructured tensor.

        Args:
            custom_shape: The shape of the original dense tensor
            compressed_tensor: A flattened tensor to store the specified elements and mask metadata.
            transposed: Whether to transpose the compressed_tensor before passing it to the cuSPARSELT.

        Returns:
            None
        """
        self.compressed_tensor = compressed_tensor
        self.transposed = transposed


    def __repr__(self):
        return (f"SparseSemiStructuredTensor(shape={self.shape} "
                f"transposed={self.transposed})")

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # since we create a new compressed tensor, the tensor will already be detached
        # this effecitvely functions as a no-op.
        if func is torch.ops.aten.detach.default:
            return SparseSemiStructuredTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. If it's an odd number of times, we'll
        # throw an error since we can't handle this situation.
        if func is torch.ops.aten.t.default:
            return SparseSemiStructuredTensor(
                args[0].shape,
                args[0].compressed_tensor,
                not args[0].transposed,
            )

        # handle mm
        if func is torch.ops.aten.mm.default:
            input_A, input_B = args

            if isinstance(input_A, cls) and not input_A.transposed:
                transposed_result, _ = torch._structured_sparse_linear(input_B.t(), input_A.values(), input_A.indices())
                return transposed_result.t()

            elif isinstance(input_B, cls) and input_B.transposed:
                result, _ = torch._structured_sparse_linear(input_A, input_B.values(), input_B.indices())
                return result

        # handle addmm
        if func is torch.ops.aten.addmm.default:
            bias, input_A, input_B = args

            # We don't support addm(bias, Sparse, Dense), because we are missing the correct bias expansion

            # Currently, we only support the first matrix being sparse for addmm/mm in cuSPARSELT and CUTLASS.
            # CUTLASS only supports the first input to be sparse for a given matmul.
            # cuSPARSELt does not have this limitation, although it appears that it uses CUTLASS under the hood,
            # since it will be slower to have the second matrix be sparse.

            # Although we only support the first matrix being sparse, we can support the second matrix being sparse.
            # We do this by taking advantage of some transpose properties:
            # F.linear(x) = addmm(bias, input, weight.t()) = b + xW' = (b + xW')''
            #        = (W''x' + b')' = (Wx' + b')' = W.cslt.addmm(input, ).T
            if isinstance(input_B, cls) and input_B.transposed:
                result, _ = torch._structured_sparse_linear(input_A, input_B.values(), input_B.indices(), bias=bias)
                return result

        # When torch is run with inference mode, pytorch does not decompose torch.ops.aten.linear into a .t() and addmm(),
        # so we must match the aten.linear op.
        # TODO see if there's a way to force pytorch to decompose the op so we don't have to handle this here.
        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            if isinstance(weight, cls):
                result, _ = torch._structured_sparse_linear(input_tensor, weight.values(), weight.indices(), bias=bias)
                return result

        # handle values
        if func is torch.ops.aten.values.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return args[0].compressed_tensor[:num_kept_elements].view(m, k // 2)

        # handle indices
        if func is torch.ops.aten.indices.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            metadata = args[0].compressed_tensor[num_kept_elements:].view(m, -1)

            # the metadata is expected to be in different datatypes for fp16/int8 respectively.
            if args[0].dtype is torch.int8:
                return metadata.view(torch.int32)
            elif args[0].dtype is torch.float16:
                return metadata.view(torch.int16)

        error_strings  = [f"func {func} with args: "]
        error_strings += [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        raise NotImplementedError("\n".join(error_strings))


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    mask=None,
    transposed=False,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of a block size given the dtype

    - torch.float32  (r, c) must be >= and a multiple of 8
    - torch.float16  (r, c) must be >= and a multiple of 16
    - torch.bfloat16 (r, c) must be >= and a multiple of 16
    - torch.int8     (r, c) must be >= and a multiple of 32

    Args::
        original_tensor (Tensor): the dense tensor to convert
        mask (Tensor): boolean mask to apply to the original tensor
        transposed (bool, optional): whether the dense tensor is transposed

    Example::
        >>> from torch.sparse import to_sparse_semi_structured
        >>> a = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,],
                              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,],
                              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,]], dtype=torch.float16, device="cuda")
        >>> a_sparse = to_sparse_semi_structured(a)
        >>> print(a_sparse)
        SparseSemiStructuredTensor(shape=torch.Size([16, 16])
         metadata=tensor([[-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [17476],
                [17476],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370],
                [-4370]], device='cuda:0', dtype=torch.int16)
         values=tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1.]], device='cuda:0',
               dtype=torch.float16))


    """
    warnings.warn(
        (
            "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
            "and will change in the near future. Please open a Github issue "
            "for features requests and see our documentation on the torch.sparse "
            "module for further information about the project."
        ),
        UserWarning,
    )
    # check device
    if not original_tensor.is_cuda:
        raise RuntimeError((f"Error original_tensor.device= {original_tensor.device} is not supported! "
                            "Only CUDA tensors are currently supported."))

    # check dim
    if original_tensor.dim() != 2:
        raise RuntimeError((f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                            "Only 2d tensors are currently supported."))

    # check dtype
    if original_tensor.dtype not in _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG:
        raise RuntimeError((f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! "
                            "dtype must be one of: {_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG}"))

    # check shape
    m, n = original_tensor.shape
    min_size = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].min_size
    if m < min_size or m % min_size or n < min_size or n % min_size:
        # TODO add padding here
        raise RuntimeError((f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                            "Both dimensions must be larger than and a multiple of {min_size}"))


    # This code calculates the size of the compressed tensor.
    # compression factor is different based on dtype
    original_size_bytes = original_tensor.nelement() * original_tensor.element_size()
    compression_factor = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].compression_factor
    compressed_size_bytes = original_size_bytes * compression_factor // 16
    compressed_size = compressed_size_bytes // original_tensor.element_size()

    compressed_tensor = torch.empty(
        (compressed_size,),
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )

    temp = torch.ones((128, n), device=original_tensor.device).to(original_tensor.dtype)
    specified = original_tensor.masked_select(mask).view(m, n // 2)
    # TODO This is a temporoary hack to get the mask in compressed form so we can store the compressed tensor.
    # In the future, we will add in a conversion function from the mask to the meta that we can use instead.
    _ , meta = torch._structured_sparse_linear(temp, specified, mask)
    # set the specified elements
    compressed_tensor[:m * n //2] = specified.view(-1)
    # set the metadata
    compressed_tensor[m * n // 2:] = meta.view(original_tensor.dtype).view(-1)


    return SparseSemiStructuredTensor(
        original_tensor.shape, compressed_tensor, transposed
    )
