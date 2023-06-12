from collections import namedtuple
import warnings

import torch


__all__ = [
    "to_sparse_semi_structured",
    "SparseSemiStructuredTensor",
]

CUSPARSELT_CONFIG = namedtuple("CUSPARELT_CONFIG", "compression_factor n min_size")
DTYPE_TO_CUSPARSELT_CONFIG = {
    torch.float32 : CUSPARSELT_CONFIG(9, 1, 8),
    torch.float16 : CUSPARSELT_CONFIG(9, 2, 16),
    torch.bfloat16 : CUSPARSELT_CONFIG(9, 2, 16),
    torch.int8: CUSPARSELT_CONFIG(10, 4, 32),
}

class SparseSemiStructuredTensor(torch.Tensor):
    """
    This class implementes semi-structured (2:4) sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is most commonly referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    compressed_tensor is a tensor that stores both the kept elemenets and the metadata mask
    These two are stored next to each other in one contiguous tensor.

    compressed tensor = [ kept elemetns of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 3 elements to be the kept elements
    The rest of the tensor is metadata.

    This tensor subclass also has a cslt object, which is set if pytorch is built with cuSPARSELt support.
    In this case, we store some additional metadata containg the sparse matrix descriptor in order for
    faster matmul performance.

    In the future, when this cslt object is not set, we can use _structured_sparse_linear.
    """

    # When _fuse_transpose is set to True, we fuse a .T into the cuSPASRELt matmul operation.
    # We do this in a bit of a hacky manner, by just creating a transposed matrix and then setting
    # the output order to be Column major instead of Row major.
    _fuse_transpose = True

    @staticmethod
    def __new__(cls, custom_shape, compressed_tensor, cslt, transposed):
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
        custom_shape,
        compressed_tensor: torch.Tensor,
        cslt,
        transposed,
    ):
        self.compressed_tensor = compressed_tensor
        self.cslt = cslt
        self.transposed = transposed


    def __repr__(self):
        return f"SparseSemiStructuredTensor(shape={self.shape} \n metadata={self.indices()} \n values={self.values()})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # since we create a new compressed tensor, the tensor will already be detached
        # this effecitvely functions as a no-op.
        if func is torch.ops.aten.detach.default:
            return SparseSemiStructuredTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. If it's an odd number of times, we'll
        # throw an error since we can't handle this situation.
        if func is torch.ops.aten.t.default:
            return SparseSemiStructuredTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                not args[0].transposed,
            )

        # handle addmm
        if func is torch.ops.aten.addmm.default:
            bias, input_A, input_B = args

            # Currently, we only support the first matrix being sparse for addmm/mm in cuSPARSELT and CUTLASS.
            # CUTLASS only supports the first input to be sparse for a given matmul.
            # cuSPARSELt does not have this limitation, although it appears that it uses CUTLASS under the hood,
            # since it will be slower to have the second matrix be sparse.
            # It may be using the same transpose trick we are using below
            if (
                isinstance(input_A, cls)
                and not input_A.transposed
            ):
                return input_A.cslt.addmm(
                    input_B, bias, not input_B.is_contiguous(), False
                )  # type: ignore[attr-defined]

            # Although we only support the first matrix being sparse, we can support the second matrix being sparse.
            # We do this by taking advantage of some transpose properties:
            # F.linear(x) = addmm(bias, input, weight.t()) = b + xW' = (b + xW')''
            #        = (W''x' + b')' = (Wx' + b')' = W.cslt.addmm(input, ).T
            elif isinstance(input_B, cls) and input_B.transposed:
                res = input_B.t().cslt.addmm(input_A.T, bias, True, cls._fuse_transpose)  # type: ignore[attr-defined]
                return res if cls._fuse_transpose else res.T

            raise NotImplementedError(
                (
                    f"func: {func} is currently not supported for ",
                    f"opA: {input_A.transposed} A: {input_A}",
                    f"opB: {input_B.transposed} B {input_B}",
                )
            )

        if func is torch.ops.aten.mm.default:
            input_A, input_B = args

            if (
                isinstance(input_A, cls)
                and not input_A.transposed
            ):
                return input_A.cslt.mm(input_B, not input_B.is_contiguous(), False)  # type: ignore[attr-defined]

            elif isinstance(input_B, cls) and input_B.transposed:
                res = input_B.t().cslt.mm(input_A.T, True, cls._fuse_transpose)  # type: ignore[attr-defined]
                return res if cls._fuse_transpose else res.T

            raise NotImplementedError(
                (
                    f"func: {func} is currently not supported for ",
                    f"opA: {input_A.transposed} A: {input_A}",
                    f"opB: {input_B.transposed} B {input_B}",
                )
            )

        # When torch is run with inference mode, it looks like pytorch does some merging in order to make linear faster.
        # The end result is that it will use this aten.linear.default op instead of decomposing into a .t() and addmm()
        # We handle this case in order to support with torch.inference_mode()
        if func is torch.ops.aten.linear.default:
            input, weight, bias = args
            if isinstance(weight, cls):
                res = weight.t().cslt.addmm(input.T, bias, True, cls._fuse_transpose)  # type: ignore[attr-defined]
                return res if cls._fuse_transpose else res.T

        # handle values
        if func is torch.ops.aten.values.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return args[0].compressed_tensor[:num_kept_elements].view(m, k // 2)

        # handle indices
        if func is torch.ops.aten.indices.default:
            m, k = args[0].shape
            num_kept_elements = m * k // 2
            return args[0].compressed_tensor[num_kept_elements:].view(m, -1).view(torch.int16)

        raise NotImplementedError(f"{func} on {args} is not implemented!")


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    transposed=False,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor. This subclass is
    responsible for overriding __torch_dispatch__ for accelerated sparse matmul.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a multiple of a block size given the dtype

    - torch.float32  (r, c) must be >= and a multiple of 8
    - torch.float16  (r, c) must be >= and a multiple of 16
    - torch.bfloat16 (r, c) must be >= and a multiple of 16
    - torch.int8     (r, c) must be >= and a multiple of 32

    Args::
        original_tensor (Tensor): the dense tensor to convert
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
    if original_tensor.dtype not in DTYPE_TO_CUSPARSELT_CONFIG:
        raise RuntimeError((f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! "
                            "dtype must be one of: {DTYPE_TO_CUSPARSELT_CONFIG}"))

    # check shape
    m, n = original_tensor.shape
    min_size = DTYPE_TO_CUSPARSELT_CONFIG[original_tensor.dtype].min_size
    if m < min_size or m % min_size or n < min_size or n % min_size:
        # TODO add padding here
        raise RuntimeError((f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                            "Both dimensions must be larger than and a multiple of {min_size}"))


    # This code calculates the size of the compressed tensor.
    # compression factor is different based on dtype
    num_bytes = original_tensor.nelement() * original_tensor.element_size()
    compression_factor = DTYPE_TO_CUSPARSELT_CONFIG[original_tensor.dtype].compression_factor
    compressed_size_bytes = num_bytes * compression_factor // 16
    compressed_size = compressed_size_bytes // original_tensor.element_size()

    compressed_tensor = torch.empty(
        (compressed_size,),
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )

    # try to use cuSPARSELt
    # TODO default to CUTLASS is cuSPARELt is not avaliable
    cslt = torch.classes.cusparselt.CusparseLt(compressed_tensor)
    # TODO Add option to prune tensor within cuSPARSELt
    cslt.compress(original_tensor, transposed)
    return SparseSemiStructuredTensor(
        original_tensor.shape, compressed_tensor, cslt, transposed
    )
