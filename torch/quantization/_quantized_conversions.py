import torch


# Transpose the input, and then reorder its elements according to
# underlying requirements of CUTLASS library, so that it could be used
# for CUTLASS-based mixed datatypes linear operation.
def quantized_weight_reorder_for_mixed_dtypes_linear(inp):
    assert inp.dim() == 2
    assert inp.dtype == torch.int8
    assert inp.device.type == "cuda"

    device = inp.device

    nrows, ncols = inp.shape
    assert nrows % 64 == 0
    assert ncols % 64 == 0

    # subbyte_transpose
    tmp = inp.T

    # permute_B_rows_for_mixed_gemm
    # (permute cols actually, as transpose is applied first here)
    cols_permuted = (
        torch.tensor(
            [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
            device=device,
        ).expand(nrows // 16, 16)
        + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
            nrows // 16, 16
        )
    ).view(-1)
    outp = tmp.index_copy(1, cols_permuted, tmp)

    # interleave_column_major_tensor
    magic0 = 2
    magic1 = 16

    tmp0 = (
        (torch.arange(0, ncols // magic0, device=device) * (nrows // 4 * magic0))
        .view(-1, 1)
        .repeat(1, nrows // 4 * magic0)
        .view(-1)
    )
    tmp1 = (
        (torch.arange(0, nrows // 4 // magic1, device=device) * (magic0 * magic1))
        .view(-1, 1)
        .repeat(1, magic1)
        .view(-1)
        .repeat(ncols)
    )
    tmp2 = (
        (torch.arange(0, magic0, device=device) * magic1)
        .view(-1, 1)
        .repeat(1, nrows // 4)
        .view(-1)
        .repeat(ncols // magic0)
    )
    tmp3 = torch.arange(0, magic1, device=device).repeat(nrows // 4 * ncols // magic1)

    outp_offsets = tmp0 + tmp1 + tmp2 + tmp3

    tmp = outp.view(-1).view(torch.int32)
    outp = torch.zeros_like(tmp)
    outp.scatter_(0, outp_offsets, tmp)
    outp = outp.view(inp.dtype).view(inp.shape)

    # add_bias_and_interleave_quantized_tensor_inplace
    tmp = outp.view(-1)

    outp = torch.empty_like(tmp)
    outp[0::4] = tmp[0::4]
    outp[1::4] = tmp[2::4]
    outp[2::4] = tmp[1::4]
    outp[3::4] = tmp[3::4]
    outp = (outp.to(torch.int) + 128).to(tmp.dtype)

    return outp.view(inp.shape)
