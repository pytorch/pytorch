import torch


class SemiStructuredSparseTensor(torch.Tensor):
    """
    This class implementes semi-structured (2:4) sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is most commonly referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    compressed_tensor is a tensor that stores both the kept elemenets and the metadata mask
    These two are stored next to each other in one contiguous tensor.

    compressed tensor = [ kept elemetns of original tensor |   mask_metadata     ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    This tensor subclass also has a cslt object, which is set if pytorch is built with cuSPARSELt support.
    In this case, we store some additional metadata containg the sparse matrix descriptor in order for
    faster matmul performance.

    When this cslt object is not set, we will default to using CUTLASS kernels and _structured_sparse_linear.
    """

    # When fuse_transpose is set to True, we fuse a .T into the cuSPASRELt matmul operation.
    # We do this in a bit of a hacky manner, by just creating a transposed matrix and then setting
    # the output order to be Column major instead of Row major.
    fuse_transpose = True

    @staticmethod
    def __new__(cls, custom_shape, compressed_tensor, cslt, transposed):
        kwargs = {}
        kwargs["device"] = compressed_tensor.device
        kwargs["dtype"] = compressed_tensor.dtype
        kwargs["layout"] = compressed_tensor.layout
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(cls, custom_shape, **kwargs)

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
        return f"SemiStructruredSparseTensor(shape={self.shape} \n kept_elements={self.kept_elements} \n metadata={self.metadata})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # since we create a new compressed tensor, the tensor will already be detached
        # this effecitvely functions as a no-op.
        if func is torch.ops.aten.detach.default:
            return SemiStructuredSparseTensor(
                args[0].shape,
                args[0].compressed_tensor,
                args[0].cslt,
                args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. If it's an odd number of times, we'll
        # throw an error since we can't handle this situation.
        if func is torch.ops.aten.t.default:
            return SemiStructuredSparseTensor(
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
            # cuSPARSELt does not have this limitation, although it appears that it uses CUTLASS under the hood, since it will be slower to have the second matrix be sparse.
            # It may be using the same transpose trick we are using below
            # input_A cannot be transposed.
            if (
                isinstance(input_A, SemiStructuredSparseTensor)
                and not input_A.transposed
            ):
                return input_A.cslt.addmm(
                    input_B, bias, not input_B.is_contiguous(), False
                )

            # Although we only support the first matrix being sparse, we can suppor the second matrix being sparse using some transpose properties.
            # F.linear(x) = addmm(bias, input, weight.t()) = b + xW' = (b + xW')'' = (W''x' + b')' = (Wx' + b')' = W.cslt.addmm(input, ).T
            elif isinstance(input_B, SemiStructuredSparseTensor) and input_B.transposed:
                res = input_B.t().cslt.addmm(input_A.T, bias, True, cls.fuse_transpose)
                return res if cls.fuse_transpose else res.T

            raise NotImplementedError(
                f"func: {func} is currently not supported for \n opA: {input_A.transposed} A: {input_A} \n opB: {input_B.transposed} B {input_B}"
            )

        if func is torch.ops.aten.mm.default:
            input_A, input_B = args

            if (
                isinstance(input_A, SemiStructuredSparseTensor)
                and not input_A.transposed
            ):
                return input_A.cslt.mm(input_B, not input_B.is_contiguous(), False)

            elif isinstance(input_B, SemiStructuredSparseTensor) and input_B.transposed:
                res = input_B.t().cslt.mm(input_A.T, True, cls.fuse_transpose)
                return res if cls.fuse_transpose else res.T

            raise NotImplementedError(
                f"func: {func} is currently not supported for \n opA: {input_A.transposed} A: {input_A} \n opB: {input_B.transposed} B {input_B}"
            )

        # When torch is run with inference mode, it looks like pytorch does some merging in order to make linear faster.
        # The end result is that it will use this aten.linear.default op instead of decomposing into a .t() and addmm()
        # We handle this case in order to support with torch.inference_mode()
        if func is torch.ops.aten.linear.default:
            input, weight, bias = args
            if isinstance(weight, SemiStructuredSparseTensor):
                res = weight.t().cslt.addmm(input.T, bias, True, cls.fuse_transpose)
                return res if cls.fuse_transpose else res.T

        raise NotImplementedError(f"{func} on {args} is not implemented!")
