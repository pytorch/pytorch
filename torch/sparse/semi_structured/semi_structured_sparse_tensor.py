import torch

from torch.utils._python_dispatch import TorchDispatchMode


class LazySparseResult(torch.Tensor):

    @staticmethod
    def __new__(
        cls,
        custom_shape,
        compressed_tensor,
        cslt,
    ):
        kwargs = {}
        kwargs["device"] = compressed_tensor.device
        kwargs["dtype"] = compressed_tensor.dtype
        # layout will be set to semi_structured_sparse eventually, but keep this strided for now
        kwargs["layout"] = torch.sparse_coo
        # currently backprop is not implented for cusparselt matmul
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(cls, custom_shape, **kwargs)

    def __init__(
        self,
        custom_shape,
        compressed_tensor: torch.Tensor,
        cslt,
        transpose_sparse, 
        transpose_dense, 
        transpose_result,
        a,
        bias,
    ):
        self.compressed_tensor = compressed_tensor
        self.cslt = cslt

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        if func is torch.ops.aten.detach.default:
            # since we create a new compressed tensor, the tensor will already be detached.
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
            )

        if func is torch.ops.aten.t.default:
            # Because we cannot go from the compressed representation back to the dense representation currently,
            # we just keep track of how many times we have been transposed. If it's an odd number of times, we'll
            # throw an error since we can't handle this situation.
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                not args[0].transposed,
            )

        if (
            func is torch.ops.aten.addmm.default
            and args[0].is_cuda
        ):
            bias, a, b = args
            if isinstance(a, SemiStructuredSparseTensor) and not a.transposed:
                # currently BIAS is broadcasted the wrong way in cusparselt, so we need to call mm and then add at the end
                return bias + a.cslt.mm(b, false, false)
            # b must be transposed so we can undo it
            elif isinstance(b, SemiStructuredSparseTensor) and b.transposed:
                # here we check if cusparselt object is set.
                res = b.t().cslt.addmm(a, bias, True, cls.fuse_transpose)
                return res if cls.fuse_transpose else res.T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, SemiStructuredSparseTensor):
                if not a.transposed:
                    return a.cslt.mm(b, False, False)
            elif isinstance(b, SemiStructuredSparseTensor):
                if b.transposed:
                    res = b.t().cslt.mm(a, True, cls.fuse_transpose)
                    return res if cls.fuse_transpose else res.T
                else:
                    return b.cslt.mm(a, True, cls.fuse_transpose)

        raise NotImplementedError(f"{func} on {args} is not implemented!")

class SemiStructuredSparseTensor(torch.Tensor):
    """
    This class implementes 2x4 sparsity as a tensor subclass.

    compressed_tensor is a tensor that stores both the kept elemenets and the metadata mask
    These two are stored next to each other in one contiguous tensor.

    compressed tensor = [ kept elemetns of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.


    This tensor subclass also has a cslt object, which is set if pytorch is built with cuSPARSELt support. 
    In this case, we store some additional metadata containg the sparse matrix descriptor in order for 
    faster matmul performance. 

    When this cslt object is not set, we default to using CUTLASS kernels and _structured_sparse_linear.

    """
    fuse_transpose = True

    @staticmethod
    def __new__(
        cls,
        custom_shape,
        compressed_tensor,
        cslt,
    ):
        kwargs = {}
        kwargs["device"] = compressed_tensor.device
        kwargs["dtype"] = compressed_tensor.dtype
        # layout will be set to semi_structured_sparse eventually, but keep this strided for now
        kwargs["layout"] = torch.sparse_coo
        # currently backprop is not implented for cusparselt matmul
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(cls, custom_shape, **kwargs)

    def __init__(
        self,
        custom_shape,
        compressed_tensor: torch.Tensor,
        cslt,
        transpose_sparse, 
        transpose_dense, 
        transpose_result,
        a,
        bias,
    ):
        self.compressed_tensor = compressed_tensor
        self.cslt = cslt

    @property
    def kept_elements(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        return self.compressed_tensor[:num_kept_elements].view(m, k // 2)

    @property
    def metadata(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        return self.compressed_tensor[num_kept_elements:].view(m, -1).view(torch.int16)

    def __repr__(self):
        return f"SemiStructruredSparseTensor(shape={self.shape})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        if func is torch.ops.aten.detach.default:
            # since we create a new compressed tensor, the tensor will already be detached.
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
            )

        if func is torch.ops.aten.t.default:
            # Because we cannot go from the compressed representation back to the dense representation currently,
            # we just keep track of how many times we have been transposed. If it's an odd number of times, we'll
            # throw an error since we can't handle this situation.
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                not args[0].transposed,
            )

        if (
            func is torch.ops.aten.addmm.default
            and args[0].is_cuda
        ):
            bias, a, b = args
            if isinstance(a, SemiStructuredSparseTensor) and not a.transposed:
                # currently BIAS is broadcasted the wrong way in cusparselt, so we need to call mm and then add at the end
                return bias + a.cslt.mm(b, false, false)
            # b must be transposed so we can undo it
            elif isinstance(b, SemiStructuredSparseTensor) and b.transposed:
                # here we check if cusparselt object is set.
                res = b.t().cslt.addmm(a, bias, True, cls.fuse_transpose)
                return res if cls.fuse_transpose else res.T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, SemiStructuredSparseTensor):
                return LazySparseResult( )
                if not a.transposed:
                    return a.cslt.mm(b, False, False)
            elif isinstance(b, SemiStructuredSparseTensor):
                if b.transposed:
                    res = b.t().cslt.mm(a, True, cls.fuse_transpose)
                    return res if cls.fuse_transpose else res.T
                else:
                    return b.cslt.mm(a, True, cls.fuse_transpose)

        raise NotImplementedError(f"{func} on {args} is not implemented!")
