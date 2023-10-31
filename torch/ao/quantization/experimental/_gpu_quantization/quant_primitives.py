import torch
from torch._dynamo import is_compiling as dynamo_is_compiling
from torch._higher_order_ops.out_dtype import out_dtype

__all__ = [
    "safe_int_mm",
    "dynamically_quantize_per_tensor",
    "quantize_activation_per_token_absmax",
    "dynamically_quantize_per_channel",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "quant_int8_dynamic_linear",
    "quant_int8_matmul",
    "quant_int8_dynamic_per_token_linear",
    "quant_int8_per_token_matmul",
]


def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    r"""
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.

    Note: no error checking for torch.compiled path, if input.shape = [i, j] and j<=16 then the triton kernel
    will error.

    Args:
        input (Tensor, int8): the first tensor to be multiplied
        mat2 (Tensor, int8): the second tensor to be multiplied

    Return:
        out (Tensor, int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
        return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

    # error checking for cublas path
    assert (
        mat2.device == input.device
    ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
    device_cpu = "cpu" in [mat2.device.type, input.device.type]
    # with input.shape = [i,j] and mat2.shape = [j,k]
    i_is_strictly_greater_than_16 = input.shape[0] > 16
    j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
    k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
    bad_dimensions_for_cublas = not (
        i_is_strictly_greater_than_16
        and j_is_nonzero_multiple_of_8
        and k_is_nonzero_multiple_of_8
    )

    if device_cpu or bad_dimensions_for_cublas:
        # fallback path
        return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
            input.device.type
        )

    # cublas paths
    if not mat2.is_contiguous():  # silently gives incorrect result without this
        mat2 = mat2.contiguous()
    if (not input.is_contiguous()) and (
        input.shape[0] % 8 != 0
    ):  # gives cryptic error without this
        input = (
            input.contiguous()
        )  # (it seems the transpose makes cublas check the above j constraint on i)
    return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)


# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
def dynamically_quantize_per_tensor(
    x,
    quant_min,
    quant_max,
    target_dtype,
    qscheme=torch.per_tensor_affine,  # for now, reuse existing qscheme enum
):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    if qscheme == torch.per_tensor_affine:
        # get min and max
        # TODO(future): make torch.aminmax work on cpu-half
        # min_val, max_val = torch.aminmax(x)
        min_val = torch.min(x)
        max_val = torch.max(x)

        # calculate scale and zero point based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # TODO(future): make torch.clamp with scalar work on cpu-half
        scale = torch.clamp(scale, min=eps).reshape(1)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # quantize based on qmin/qmax/scale/zp
        # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
        quant = torch.clamp(
            torch.round(x / scale) + zero_point, quant_min, quant_max
        ).to(target_dtype)

    else:
        assert qscheme == torch.per_tensor_symmetric, f"unsupported qscheme {qscheme}"
        # assert quant_min == -1 * quant_max, "unsupported quant_min/quant_max"
        amax = torch.max(torch.abs(x))
        scale = amax / (float(quant_max - quant_min) / 2)
        scale = torch.clamp(scale, min=eps).reshape(1)
        quant = torch.clamp(torch.round(x / scale), quant_min, quant_max).to(
            target_dtype
        )
        # do not create a tensor for zero_point as this is expensive
        zero_point = None

    return quant, scale, zero_point


# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]

    scales = t.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point


# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    y = int_repr.to(out_dtype)
    if zero_point is not None:
        y -= zero_point
    return y * scale


# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    # assumes axis is 0
    y = int_repr.transpose(0, 1)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales
    y = y.transpose(0, 1)
    return y


def quant_int8_dynamic_linear(
    x,
    x_quant_min,
    x_quant_max,
    x_q_dtype,
    w_vals_int8_t,
    w_scales,
    w_vals_int8_t_sums_int64,
    bias,
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype
    )
    # w_vals_int8_t_sums_int64 = w_vals_int8_t.sum(dim=0)
    mm_out = quant_int8_matmul(
        x_vals_int8,
        x_scale,
        x_zp,
        w_vals_int8_t,
        w_vals_int8_t_sums_int64,
        w_scales,
        out_dtype,
    )
    if bias is not None:
        mm_out += bias
    return mm_out


def quant_int8_matmul(
    x_vals_int8,
    x_scale,
    x_zp,
    w_vals_int8_t,
    w_vals_int8_t_sums_int64,
    w_scales,
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw (dot(X_i, W_j) - zx * sum(W_j))
    #

    assert x_vals_int8.dtype in (
        torch.uint8,
        torch.int8,
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"
    assert w_scales.dtype == out_dtype, f"{w_scales.dtype} does not match {out_dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)

    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    y_dot_float32 = y_dot_int32.to(torch.float32)

    #
    # 2. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = y_dot_float32 - (x_zp * w_vals_int8_t_sums_int64)
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y


def quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype=torch.float32,
    use_fused_int_mm=0,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype, use_fused_int_mm
    )
    if bias is not None:
        mm_out += bias
    return mm_out


def quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    output_dtype=torch.float32,
    use_fused_int_mm=0,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # output_dtype. For now, this is written for approximate numerical
    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming output_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"
    assert (
        w_scales.dtype == output_dtype
    ), f"{w_scales.dtype} does not match {output_dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    # these branches use external triton fused_int_mm kernel's which fuse either 1 or 2 mul operations
    if use_fused_int_mm == 2:
        y = torch.ops.custom_int_mm.int_mm_dequant(
            tmp, w_vals_int8_t, x_scales.view(-1, 1), w_scales, output_dtype
        ).reshape(*x_vals_int8.shape[:-1], -1)
        return y
    elif use_fused_int_mm == 1:
        y = torch.ops.custom_int_mm.int_mm_one_mul(
            tmp, w_vals_int8_t, x_scales.view(-1, 1), output_dtype
        ).reshape(*x_vals_int8.shape[:-1], -1)
        y = y * w_scales
        return y.to(output_dtype)
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)

    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"
    y = (y_dot_int32 * x_scales.view(-1, 1) * w_scales).reshape(
        *x_vals_int8.shape[:-1], -1
    )

    # can downcast only at the very end
    y = y.to(output_dtype)
    return y
