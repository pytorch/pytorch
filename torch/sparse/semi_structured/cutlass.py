import torch
from torch.sparse import SparseSemiStructuredTensor

__all__ = ["SparseSemiStructuredTensorCUTLASS"]

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor, torch.Tensor, metaclass=SparseSemiStructuredMeta):
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
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4)
    }

    @staticmethod
    def __new__(
        cls,
        sparse_tensor_cutlass: torch.Tensor,
        meta_tensor_cutlass: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
        transposed: bool = False,
    ) -> torch.Tensor:
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

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[torch.Size, bool]]:
        return ['sparse_tensor_cutlass', 'meta_tensor_cutlass'], (self.original_shape, self.transposed)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride) -> SparseSemiStructuredTensorCUTLASS:
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

    @classmethod
    def from_dense(cls, original_tensor) -> SparseSemiStructuredTensorCUTLASS:
        cls._validate_device_dim_dtype_shape(original_tensor)
        sparse_tensor_cutlass, meta_tensor_cutlass = sparse_semi_structured_from_dense_cutlass(original_tensor)
        return cls(sparse_tensor_cutlass, meta_tensor_cutlass, original_shape=original_tensor.shape)

    def to_dense(self) -> torch.Tensor:
        if self.sparse_tensor_cutlass.dtype == torch.float32:
            raise RuntimeError("Converting to dense for torch.float32 datatype is not yet supported by CUTLASS backend!")

        return sparse_semi_structured_to_dense_cutlass(
            self.sparse_tensor_cutlass,
            self.meta_tensor_cutlass,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:

        if func is torch.ops.aten.values.default:
            return args[0].sparse_tensor_cutlass.detach()

        if func is torch.ops.aten.indices.default:
            return args[0].meta_tensor_cutlass

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
                res = torch._sparse_semi_structured_linear(
                    input_B_padded.t(),
                    input_A.sparse_tensor_cutlass,
                    input_A.meta_tensor_cutlass,
                    bias=bias
                ).t()
                return res[:, :col]

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

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )

        raise NotImplementedError(error_string)

def _sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float32:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1101
    #     [False, True,  True,  True ] -> 0b1001
    #     [True,  False, False, False] -> 0b1100
    #     [True,  False, True,  True ] -> 0b1000
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b1000
    #     [True,  True,  True,  True ] -> 0b1000
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float32:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # Metadata values are now to be reshuffled in a way given in
    # reorder_meta() function, in
    # tools/util/include/cutlass/util/host_reorder.h file of CUTLASS
    # source tree.  Furthermore, CUTLASS template for sparse GEMM
    # decides upon layout of this matrix, and at the moment for the
    # sparse GEMM executed on tensor cores, this is layout described
    # by ColumnMajorInterleaved<2> data structure, in
    # include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    # reordering of meta matrix into meta_reordered matrix calculated
    # according to these segments of CUTLASS code is given below.
    # However, this calculation produces offsets for scatter access
    # from metadata matrix to redordered metadata matrix, and gather
    # pattern is more efficient.  For this reason, the scatter offsets
    # are reverted and printed, through enabling commented block at
    # the end of following code.  Resulting gather offsets are then
    # analyzed, on several (m, k) value pairs (in particular: (32,
    # 128), (32, 256), (64, 128) and (64, 256)), and the code that
    # follows this comment is written to reproduce these gather offsets.
    #
    #    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    #    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)
    #
    #    # Reorder the rows, then swizzle the 2x2 blocks.
    #    group = 32 if meta_dtype.itemsize == 2 else 16
    #    interweave = 4 if meta_dtype.itemsize == 2 else 2
    #    dst_rows = (
    #        dst_rows // group * group
    #        + (dst_rows % 8) * interweave
    #        + (dst_rows % group) // 8
    #    )
    #
    #    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    #    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    #    dst_rows += topright - bottomleft
    #    dst_cols -= topright - bottomleft
    #
    #    # Assumed that meta tensor is to be stored in CUTLASS
    #    # InterleavedColumnMajor layout, and reverse engineered
    #    # corresponding code to store values into this tensor.
    #    interleave = 2
    #    cols_maj = dst_cols // interleave
    #    cols_min = dst_cols % interleave
    #    meta_reordered_offsets = (
    #        cols_maj * m * interleave + dst_rows * interleave + cols_min
    #    )
    #
    #    meta_reordered = torch.empty((m, meta_ncols), dtype=meta_dtype, device=device)
    #    meta_reordered.view(-1)[meta_reordered_offsets.view(-1)] = meta.view(-1)
    #
    #    # Uncomment to have gather pattern for meta_reordered printed
    #    #
    #    #offsets = torch.empty(
    #    #    (m, meta_ncols), dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #offsets.view(-1)[meta_reordered_offsets.view(-1)] = torch.arange(
    #    #    0, m * meta_ncols, dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #torch.set_printoptions(threshold=1000000)
    #    #print("------------------------------------------------------------")
    #    #print("dtype =", dtype, ", m =", m, ", k =", k, ", meta_ncols =", meta_ncols)
    #    #print(offsets.view(-1))
    #

    # No point to try to understand this code: as mentioned in the
    # comment above it is written to reproduce gather offsets, as
    # these would be calculated by CUTLASS, and to be efficient, but
    # it contains several magic values and magic calculations that
    # make it rather hard to read, let alone understand.
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = 2 * k // ksparse
        magic4 = [0, k // ksparse, 1, k // ksparse + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 4 * 2 * k // ksparse
        magic4 = [0, 2 * k // ksparse, 1, 2 * k // ksparse + 1, 2 * 2 * k // ksparse,
                  3 * 2 * k // ksparse, 2 * 2 * k // ksparse + 1, 3 * 2 * k // ksparse + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (
        tmp0.view(meta_ncols // 2, -1)
        + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)
    ).view(-1, magic1)
    tmp2 = (
        (
            torch.arange(0, 8, device=device).view(-1, 1)
            * torch.ones((magic0,), dtype=torch.int64, device=device)
            * meta_ncols
        )
        .view(-1)
        .repeat(m * meta_ncols // magic1)
        .view(-1, magic1)
    )
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(
        meta_ncols // 2, magic1
    )
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4

    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )

    return (sparse, meta_reordered)


def _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(
        meta_nrows, meta_ncols // 2
    )
    tmp2 = (
        (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows)
        .view(-1, 1)
        .repeat(1, 2)
        .view(-1)
        .repeat(m, 1)
    )
    tmp3 = (
        (torch.arange(0, 8, device=device) * magic0)
        .view(-1, 1)
        .repeat(m // 8, meta_ncols)
    )
    tmp4 = (
        torch.tensor(magic1, device=device)
        .view(-1, 1)
        .repeat(1, 8 * meta_ncols)
        .repeat(meta_nrows // 32, 1)
        .view(meta_nrows, meta_ncols)
    )
    tmp5 = (
        (torch.arange(0, meta_nrows // 32, device=device) * 64)
        .view(-1, 1)
        .repeat(1, 32 * meta_ncols)
        .view(meta_nrows, meta_ncols)
    )
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5

    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )

    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, m * k // 2, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))

    return dense.view(m, 2 * k)


# This function converts dense matrix into sparse semi-structured
# representation, producing "compressed" matrix, in the layout used by
# CUTLASS backend, and corresponding metadata matrix.
def sparse_semi_structured_from_dense_cutlass(dense, compile=False):
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(dense.device.type):
            kernel = torch.compile(_sparse_semi_structured_from_dense_cutlass)
            return kernel(dense)

    return _sparse_semi_structured_from_dense_cutlass(dense)


# This function performs reverse of the function above - it
# reconstructs dense matrix from a pair of "compressed" matrix, given
# in the layout used by CUTLASS backend, and accompanying metadata
# matrix.
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered, compile=False):
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(sparse.device.type):
            kernel = torch.compile(_sparse_semi_structured_to_dense_cutlass)
            return kernel(sparse, meta_reordered)

    return _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered)
