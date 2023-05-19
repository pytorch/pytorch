import torch


class SemiStructuredSparseTensor(torch.Tensor):
    """
    This class implementes 2x4 sparsity
    """

    @staticmethod
    def __new__(
        cls,
        original_tensor,
        original_shape=None,
        compressed_tensor=None,
        cslt=None,
        transposed=False,
        contiguous_output=False,
    ):
        kwargs = {}
        kwargs["device"] = (
            original_tensor.device
            if original_tensor is not None
            else compressed_tensor.device
        )
        kwargs["dtype"] = (
            original_tensor.dtype
            if original_tensor is not None
            else compressed_tensor.dtype
        )
        kwargs["layout"] = (
            original_tensor.layout
            if original_tensor is not None
            else compressed_tensor.layout
        )
        kwargs["requires_grad"] = (
            original_tensor.requires_grad if original_tensor is not None else False
        )

        shape = original_shape if original_shape is not None else original_tensor.shape

        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        original_tensor,
        original_shape=None,
        compressed_tensor=None,
        cslt=None,
        transposed=False,
        contiguous_output=False,
    ):
        self.original_tensor = original_tensor
        self.original_shape = (
            original_tensor.shape if original_shape is None else original_shape
        )
        self.compressed_tensor = compressed_tensor
        self.cslt = cslt
        self.transposed = transposed
        self.contiguous_output = contiguous_output

        if original_tensor is not None:
            num_bytes = original_tensor.nelement() * original_tensor.element_size()
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

            self.cslt = torch.classes.cusparselt.CusparseLt(self.compressed_tensor)
            self.cslt.compress(original_tensor, self.transposed)
            self.original_tensor = None

    def __repr__(self):
        return f"SemiStructruredSparseTensor(shape={self.shape})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func is torch.ops.aten.detach.default:
            # since we create a new compressed tensor, the tensor will already be detached.
            return SemiStructuredSparseTensor(
                args[0].original_tensor,
                compressed_tensor=args[0].compressed_tensor,
                cslt=args[0].cslt,
                original_shape=args[0].original_shape,
                transposed=args[0].transposed,
                contiguous_output=args[0].contiguous_output,
            )

        if func is torch.ops.aten.t.default:
            # Because we cannot go from the compressed representation back to the dense representation currently, we just keep track of how many times we have been transposed. If it's an odd number of times, we'll throw an error since we can't handle this situation
            return SemiStructuredSparseTensor(
                args[0].original_tensor,
                compressed_tensor=args[0].compressed_tensor,
                cslt=args[0].cslt,
                original_shape=args[0].original_shape,
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
