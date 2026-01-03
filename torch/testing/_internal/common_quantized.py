# mypy: ignore-errors

r"""Importing this file includes common utility methods for checking quantized
tensors and modules.
"""
import numpy as np
import torch
from torch import Tensor
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_TSAN, IS_PPC, IS_MACOS, IS_WINDOWS, IS_ARM64

supported_qengines = list(torch.backends.quantized.supported_engines)
# Note: We currently do not run QNNPACK tests on WINDOWS and MACOS as it is flaky. Issue #29326
# QNNPACK is not supported on PPC
if 'qnnpack' in supported_qengines and any([IS_PPC, TEST_WITH_TSAN, IS_MACOS, IS_WINDOWS]):
    supported_qengines.remove('qnnpack')
# FBGEMM and x86 engines require x86 architecture with AVX2/AVX512 support
# They are not supported on ARM64 architectures
if IS_ARM64:
    supported_qengines = [qe for qe in supported_qengines if qe not in ('fbgemm', 'x86')]

def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    """Computes the output shape given convolution parameters."""
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1)
                     * (dilation - 1)) / stride) + 2 * output_padding + 1

# Quantization references
def _quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=np.uint8):
    """Quantizes a numpy array."""
    if qmin is None:
        qmin = np.iinfo(dtype).min
    if qmax is None:
        qmax = np.iinfo(dtype).max
    qx = np.round(x / scale + zero_point).astype(np.int64)
    qx = np.clip(qx, qmin, qmax)
    qx = qx.astype(dtype)
    return qx


def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = (qx.astype(float) - zero_point) * scale
    return x


def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """Requantizes a numpy array, i.e., intermediate int32 or int16 values are
    converted back to given type"""
    qx = (x * multiplier).round() + zero_point
    qx = np.clip(qx, qmin, qmax).astype(qtype)
    return qx

def _calculate_dynamic_qparams(X, dtype, reduce_range=False, qscheme=torch.per_tensor_affine):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    assert qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    if qscheme == torch.per_tensor_symmetric:
        assert dtype == torch.qint8
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255
    min_val = X.min()
    max_val = X.max()
    is_symmetric = (qscheme == torch.per_tensor_symmetric)
    if min_val == max_val:
        scale = 1.0
        zero_point = 0
    else:
        if is_symmetric:
            max_val = max(max_val, -min_val)
            min_val = -max_val
            scale = (max_val - min_val) / (qmax - qmin)
            scale = max(scale, np.finfo(np.float32).eps)
            zero_point = 0
        else:
            max_val = max(max_val, 0.0)
            min_val = min(min_val, 0.0)
            scale = (max_val - min_val) / (qmax - qmin)
            scale = max(scale, np.finfo(np.float32).eps)
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
    return [float(scale), int(zero_point)]

def _calculate_dynamic_per_channel_qparams(X, dtype):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    n_levels = qmax - qmin
    scale = np.zeros(X.shape[0], dtype=np.float64)
    zero_point = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(zero_point.shape[0]):
        min_val = X.min()
        max_val = X.max()
        if min_val == max_val:
            scale[i] = 1.0
            zero_point[i] = 0
        else:
            max_val = max(max_val, 0.0)
            min_val = min(min_val, 0.0)
            scale[i] = (max_val - min_val) / n_levels
            scale[i] = max(scale[i], np.finfo(np.float32).eps)
            zero_point[i] = qmin - round(min_val / scale[i])
            zero_point[i] = max(qmin, zero_point[i])
            zero_point[i] = min(qmax, zero_point[i])

    return scale, zero_point

def _snr(x, x_hat):
    """Calculates the signal to noise ratio and returns the signal and noise
    power, as well as the SNR in dB.
    If the input is a list/tuple this function is called recursively on each
    element. The result will have the same nested structure as the inputs.

    Args:
        x, x_hat: Either a tensor or a nested list/tuple of tensors.
    Returns:
        signal, noise, SNR(in dB): Either floats or a nested list of floats
    """
    if isinstance(x, (list, tuple)):
        assert len(x) == len(x_hat)
        res = [_snr(x[idx], x_hat[idx]) for idx in range(len(x))]
        return res
    if x_hat.is_quantized:
        x_hat = x_hat.dequantize()
    if x.is_quantized:
        x = x.dequantize()
    noise = (x - x_hat).norm()
    if noise == 0:
        return 0.0, float('inf'), float('inf')
    signal = x.norm()
    snr = signal / noise
    snr_db = 20 * snr.log10()
    return signal, noise, snr_db

@contextmanager
def override_quantized_engine(qengine):
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = qengine
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous

@contextmanager
def override_cpu_allocator_for_qnnpack(qengine_is_qnnpack):
    try:
        if qengine_is_qnnpack:
            torch._C._set_default_mobile_cpu_allocator()
        yield
    finally:
        if qengine_is_qnnpack:
            torch._C._unset_default_mobile_cpu_allocator()

# TODO: Update all quantization tests to use this decorator.
# Currently for some of the tests it seems to have inconsistent params
# for fbgemm vs qnnpack.
def override_qengines(qfunction):
    def test_fn(*args, **kwargs):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # qfunction should not return anything.
                qfunction(*args, **kwargs)
    return test_fn

def qengine_is_fbgemm():
    return torch.backends.quantized.engine == 'fbgemm'
def qengine_is_qnnpack():
    return torch.backends.quantized.engine == 'qnnpack'
def qengine_is_onednn():
    return torch.backends.quantized.engine == 'onednn'
def qengine_is_x86():
    return torch.backends.quantized.engine == 'x86'

# Helper function used to simulate per-channel fake-quant against any axis
def _permute_to_axis_zero(X, axis):
    new_axis_list = list(range(X.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = X.permute(tuple(new_axis_list))
    return y, new_axis_list

# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_reference(X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    res = torch.zeros_like(X)

    for i in range(X.size()[0]):
        res[i] = (torch.clamp(torch.round(X[i] * (1.0 / per_channel_scale[i]) +
                  per_channel_zero_point[i]), quant_min, quant_max) - per_channel_zero_point[i]) * per_channel_scale[i]

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

# Reference method for the gradient of the fake quantize operator
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

def to_tensor(X, device):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    else:
        X = X.detach().clone()
    return X.to(device=torch.device(device), dtype=torch.float32)

# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _n_ones(n: int) -> int:
    return (1 << n) - 1

EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


# copy-pasted from
# https://github.com/pytorch/ao/blob/29488018d99af7f7339f06353c6b5bbeae8a1493/torchao/prototype/custom_fp_utils.py#L147
def _floatx_unpacked_to_f32(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp - (1 << i)) << (
                    left_shift + MBITS_F32 - mbits
                )
                exp_biased_f32 = (denormal_exp_biased - left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s) for |: 'SymInt' and 'int'
                # thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32
                )

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)

# copied from https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/mx/to_blocked.py
def ceil_div(a, b):
    return (a + b - 1) // b

# NVIDIA Blackwell HW requires scales for MX/NV blocked formats to be in a 128x4 tile layout,
# with a weird 32x4x4 internal layout of that tile. If we want to take swizzled scales and use them
# for non-gemm purposes (like testing), we need to de-swizzle them, then they can be applied much
# more naturally.
def from_blocked(input, input_scales, blocksize) -> torch.Tensor:
    # Matrix is in a 128x4 pattern, internally blocked as 32x4x4 nonsense.
    # Output should be [input.size(0, input.size(1) // blocksize] scales
    output_scales = torch.zeros(
        (input.size(0), input.size(1) // blocksize),
        device=input.device,
        dtype=input_scales.dtype,
    )

    # Swizzled scales are padded to tiles of 128x4, we need to replicate how that padding
    # happened for offset purposes.
    # There are K//blocksize scales, padded to groups of 4.
    num_col_tiles = ceil_div(ceil_div(input.size(1), blocksize), 4)

    # (Very) slow reference implementation using horrifying loops.
    for i in range(input.size(0)):
        for j in range(input.size(1) // blocksize):
            # which 128x4 tile of scaling factors am I in
            scale_tile_h = i // 128
            scale_tile_w = j // 4

            # There are (padded) input_scales.size(1) // 4 tiles along the w dim.
            # So offset is 512 * (h_tile * tiles_per_row + tile_in_row)
            tile_offset = 512 * (scale_tile_h * num_col_tiles + scale_tile_w)

            # indices within the tile - use nomenclature directly from cublas docs
            outer = i % 128  # "outer" in cublas docs
            inner = j % 4    # "inner" in cublas docs

            # Note: "offset" is given in terms of bytes, in cublas docs, but our scales are e8m0,
            #       anyway, and so 1B == 1 value => use offset directly.
            # Formula directly from cublas docs in 3.1.4.3.2
            offset = tile_offset + (outer % 32) * 16 + (outer // 32) * 4 + inner

            output_scales[i, j] = input_scales[offset]

    return output_scales

def from_blocked_format(x_mxfp8, scales_unswizzled, blocksize=32):
    # expand scales
    scales = torch.repeat_interleave(scales_unswizzled, blocksize, dim=1)

    # de-scale and convert
    x_f32 = x_mxfp8.to(torch.float) * scales.to(torch.float)
    return x_f32.to(torch.bfloat16)

def to_blocked(input_matrix) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Ideally we would use torch.nn.pad but it doesn't support float8_e8m0fnu for now
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1


def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


# This function is extracted from https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142
def to_mxfp(
    data_hp: torch.Tensor,
    block_size: int = 32,
    format: str = "mxfp8",
):
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert (
        data_hp.shape[-1] % block_size == 0
    ), f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
    assert data_hp.is_contiguous(), "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if format == "mxfp8":
        F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        max_pos = F8E4M3_MAX
    elif format == "mxfp4":
        F4E2M1_MAX = 6.
        max_pos = F4E2M1_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)),
                    min=-E8M0_EXPONENT_BIAS,
                    max=E8M0_EXPONENT_BIAS,
                )
                + E8M0_EXPONENT_BIAS
            ).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    if format == "mxfp8":
        data_lp = data_lp.to(torch.float8_e4m3fn)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    elif format == "mxfp4":
        data_lp = _bfloat16_to_float4_e2m1fn_x2(data_lp.to(torch.bfloat16))
        final_shape = list(orig_shape)
        final_shape[-1] //= 2
        data_lp = data_lp.reshape(final_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp

# Source: https://github.com/pytorch/ao/blob/568c1932a16ae9f30d48da214a88dc0013e98ed8/torchao/prototype/moe_training/utils.py#L310
def generate_jagged_offs(E, M, multiple_of=16, dtype=torch.int32, device="cuda"):
    """
    Utility function for tests and benchmarks.

    Generates a tensor of length E, containing random values divisible by `multiple_of`,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    import random

    # Ensure M is divisible by 16
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")

    # Generate a list of possible values
    possible_values = list(range(multiple_of, M + 1, multiple_of))

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = torch.tensor(random.sample(possible_values[:-1], E - 1))

    # Append M to the selected values
    selected_values = torch.cat((selected_values, torch.tensor([M])))

    # Sort the selected values
    selected_values, _ = torch.sort(selected_values)

    return selected_values.to(dtype).to(device)
