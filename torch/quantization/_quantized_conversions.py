import torch


# Pack pairs of int4 values into int8, in row major order; first int4
# value goes into lower order bits, and second int4 value into higher
# order bits of resulting int8 value.
def pack_int4_to_int8(weight):
    assert weight.dim() == 2
    assert weight.shape[1] % 2 == 0
    assert weight.dtype == torch.int8
    return ((weight[:, 1::2] & 0xF) << 4) | (weight[:, 0::2] & 0xF)


# Unpack quandruples of bits in int8 values into int4 values, in row
# major order; lower 4 bits go into first int4 value goes, and upper 4
# bits go into second int4 value.
def unpack_int8_to_int4(weight):
    assert weight.dim() == 2
    assert weight.dtype == torch.int8
    return torch.stack((weight & 0xF, (weight >> 4) & 0xF), dim=2).view(
        weight.shape[0], 2 * weight.shape[1]
    )


# Transpose the weight matrix, and then reorder its elements according
# to underlying requirements of CUTLASS library, so that it could be
# used for CUTLASS-based mixed datatypes linear operation.
def quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
    weight, dtypeq, transpose=False
):
    assert weight.dim() == 2
    assert weight.dtype == torch.int8
    assert dtypeq == torch.int8 or dtypeq == torch.quint4x2
    assert weight.device.type == "cuda"

    device = weight.device

    # subbyte_transpose
    if not transpose:
        if dtypeq == torch.int8:
            outp = weight.T
        elif dtypeq == torch.quint4x2:
            outp = pack_int4_to_int8(unpack_int8_to_int4(weight.view(torch.int8)).T)
    else:
        outp = weight

    ncols, nrows = outp.shape
    assert nrows % (32 if dtypeq == torch.quint4x2 else 64) == 0
    assert ncols % 64 == 0

    # permute_B_rows_for_mixed_gemm
    # (permute cols actually, as transpose is applied first here)
    if dtypeq == torch.quint4x2:
        cols_permuted = (
            torch.tensor(
                [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)
    else:
        cols_permuted = (
            torch.tensor(
                [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)
    outp = outp.index_copy(1, cols_permuted, outp)

    # interleave_column_major_tensor
    magic0 = 4 if dtypeq == torch.quint4x2 else 2
    magic1 = 32 // magic0

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
    outp = outp.view(weight.dtype)

    # add_bias_and_interleave_quantized_tensor_inplace
    tmp = outp.view(-1)

    outp = torch.empty_like(tmp)
    if dtypeq == torch.int8:
        tmp = (tmp.to(torch.int) + 128).to(tmp.dtype)
        outp[0::4] = tmp[0::4]
        outp[1::4] = tmp[2::4]
        outp[2::4] = tmp[1::4]
        outp[3::4] = tmp[3::4]
    elif dtypeq == torch.quint4x2:
        tmp0 = ((tmp & 0xF) + 8) & 0xF
        tmp0 = (tmp0[1::2] << 4) | tmp0[0::2]
        tmp1 = (((tmp >> 4) & 0xF) + 8) & 0xF
        tmp1 = (tmp1[1::2] << 4) | tmp1[0::2]
        outp[0::4] = tmp0[0::2]
        outp[1::4] = tmp0[1::2]
        outp[2::4] = tmp1[0::2]
        outp[3::4] = tmp1[1::2]

    if dtypeq == torch.quint4x2:
        nrows *= 2
        ncols //= 2

    return outp.view(nrows, ncols).view(torch.uint8)
