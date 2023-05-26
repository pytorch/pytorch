import torch
from torch.utils.__python_dispatch import TorchDispatchMode

def to_semi_structured_sparse(original_tensor: torch.Tensor, backend="cusparselt", transposed=False):
    # This code calculates the size of the compressed tensor. 

    num_bytes = original_tensor.nelement() * original_tensor.element_size()

    # compression factor is different based on dtype
    if original_tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        compression_factor = 9
    elif original_tensor.dtype is torch.int8:
        compression_factor = 10

    compressed_size_bytes = num_bytes * compression_factor // 16
    compressed_size = compressed_size_bytes // original_tensor.element_size()

    self.compressed_tensor = torch.empty(
        (compressed_size,),
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )

    cslt = torch.classes.cusparselt.CusparseLt(self.compressed_tensor)
    # TODO is there a better way to check this? 
    # if not contiguous -> assume is transposed
    cslt.compress(original_tensor, original_tensor.is_contiguous())

    return SemiStructuredSparseTensor(original_tensor.shape, compressed_tensor, cslt, transposed)



def from_semi_structured_sparse(sparse_tensor):
    raise NotImplementedError("Currently not supported")


class SemiStructuredSparseTensor(torch.Tensor):
    """
    This class implementes 2x4 sparsity as a tensor subclass. 

    compressed_tensor is a tensor that stores both the kept elemenets and the metadata mask
    These two are stored next to each other in one contiguous tensor. 

    compressed tensor = [ kept elemetns of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. 

    """

    @staticmethod
    def __new__(
        cls,
        custom_shape,
        cslt_compressed_tensor,
        cslt,
        transposed,
    ):
        kwargs = {}
        kwargs["device"] = compressed_tensor.device
        kwargs["dtype"] = compressed_tensor.dtype
        kwargs["layout"] = "semi_structured_sparse"
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(cls, custom_shape, **kwargs)

    def __init__(
        self,
        custom_shape,
        compressed_tensor : torch.Tensor,
        cslt,
        transposed : bool,
    ):
        self.compressed_tensor = compressed_tensor
        self.cslt = cslt
        self.transposed = transposed


    @property
    def kept_elements(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        return self.compressed_tensor[:num_kept_elements].view(m, k // 2)
   
    @property
    def metadata(self):
        m, k = self.shape
        num_kept_elements = m * k // 2
        return self.compressed_tensor[num_kept_elements:].view(m, -1)


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
                transposed=args[0].transposed,
                contiguous_output=args[0].contiguous_output,
            )

        if func is torch.ops.aten.t.default:
            # Because we cannot go from the compressed representation back to the dense representation currently, we just keep track of how many times we have been transposed. If it's an odd number of times, we'll throw an error since we can't handle this situation
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                transposed=not args[0].transposed,
                contiguous_output=args[0].contiguous_output,
            )

        if (
            func is torch.ops.aten.addmm.default
            and args[0].is_floating_point()
            and args[0].is_cuda
        ):
            bias, a, b = args
            if isinstance(a, SemiStructuredSparseTensor) and not a.transposed:
                # currently BIAS is broadcasted the wrong way in cuSPARSELT, so we need to call mm and then add at the end
                return bias + a.cslt.mm(b, False)
            # b must be transposed so we can undo it
            elif isinstance(b, SemiStructuredSparseTensor) and b.transposed:
                result, meta = torch._structured_sparse_linear(a, b.kept_elements, b.metadata, bias=bias)
                res = b.t().cslt.addmm(a.T, bias, b.contiguous_output)
                return res if b.contiguous_output else res.T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, SemiStructuredSparseTensor) and not a.transposed:
                return a.cslt.mm(b, False)
            elif isinstance(b, SemiStructuredSparseTensor) and b.transposed:
                res = b.t().cslt.mm(a.T, b.contiguous_output)
                return res if b.contiguous_output else res.T

        raise NotImplementedError("Not implemented")
