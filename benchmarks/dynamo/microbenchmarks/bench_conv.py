import model

import torch
import triton
import triton.language as tl

# import torch._inductor.triton_ops
from triton.testing import do_bench

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def conv_heuristics():
    configs = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 16}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 16}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 1024, "BLOCK_N": 16, "BLOCK_K": 16}, num_stages=1, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8
        ),
    ]
    key = [
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "OUT_C",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        "STRIDE_H",
        "STRIDE_W",
        "PADDING_H",
        "PADDING_W",
    ]
    return triton.autotune(configs, key)


@conv_heuristics()
@triton.jit
def _kernel(
    X,
    W,
    Y,
    # stride of tensor
    stride_xn: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xw: tl.constexpr,
    stride_wc_out: tl.constexpr,
    stride_wc_in: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_ww: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_yc: tl.constexpr,
    stride_yh: tl.constexpr,
    stride_yw: tl.constexpr,
    # Tensor dimensions
    BATCH: tl.constexpr,
    IN_C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_C: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    # parameters of conv
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    GROUPS: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if GROUPS > 1:
        group = tl.program_id(2)
        GROUP_IN_C = IN_C // GROUPS
        GROUP_OUT_C = OUT_C // GROUPS
    else:
        group = 0
        GROUP_IN_C = IN_C
        GROUP_OUT_C = OUT_C

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)

    mask_y = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    y_ptrs = Y + (
        idx_n[:, None] * stride_yn
        + idx_y_h[:, None] * stride_yh
        + idx_y_w[:, None] * stride_yw
        + (idx_y_c[None, :] + group * GROUP_OUT_C) * stride_yc
    )
    tl.store(y_ptrs, acc, mask=mask_y)


class _conv_analytic:
    kernel = _kernel

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layeout
        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (
            not shape_bias or shape_bias[0] == shape_w[wn]
        ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
        assert (
            shape_x[xc] == in_channel
        ), f"in_channel did not match {shape_x[xc]} != {in_channel}"
        # assert kernel_size == [3, 3], "should be 3x3 kernel"

        assert bias is None

        assert (
            len(stride)
            == len(padding)
            == len(dilation)
            == len(output_padding)
            == len(kernel_size)
            == len(input_size)
        )

        # output shape
        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (
            input_size[0]
            + 2 * padding[0]
            - dilation[0] * (kernel_size[0] - 1)
            - 1
            + stride[0]
        ) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (
            input_size[1]
            + 2 * padding[1]
            - dilation[1] * (kernel_size[1] - 1)
            - 1
            + stride[1]
        ) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        OUT_C = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        # allocate output
        y = torch.empty(shape_y, device=device, dtype=x.dtype)

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()

        # output layout should be the same as x
        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            y = y.to(memory_format=torch.channels_last)

        stride_y = y.stride()

        assert not transposed
        assert output_padding == (0, 0)
        assert dilation == (1, 1)

        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(OUT_C, META["BLOCK_N"]),
                groups,
            )

        _kernel[grid](
            x,
            w,
            y,
            # stride nchw for x,w,y tensor
            stride_x[xn],
            stride_x[xc],
            stride_x[xh],
            stride_x[xw],
            stride_w[wn],
            stride_w[wc],
            stride_w[wh],
            stride_w[ww],
            stride_y[yn],
            stride_y[yc],
            stride_y[yh],
            stride_y[yw],
            # Tensor dimensions
            BATCH,
            IN_C,
            IN_H,
            IN_W,
            OUT_C,
            OUT_H,
            OUT_W,
            # conv parameters
            KERNEL_H,
            KERNEL_W,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            groups,
            ALLOW_TF32=torch.backends.cudnn.allow_tf32,
        )
        return y

    @staticmethod
    def forward(
        x,
        w,
        bias,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    ):
        if transposed:
            print("Do not support transposed")
        return _conv_analytic._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv_analytic = _conv_analytic.forward

conv_confs = [
    {
        "BATCH": BATCH,
        "IN_H": IN_H,
        "IN_W": IN_W,
        "IN_C": IN_C,
        "KERNEL_N": KERNEL_N,
        "KERNEL_H": KERNEL_H,
        "KERNEL_W": KERNEL_W,
        "stride": stride,
        "padding": padding,
    }
    for i, (
        IN_H,
        IN_W,
        IN_C,
        KERNEL_H,
        KERNEL_W,
        KERNEL_N,
        stride,
        padding,
    ) in enumerate(model.resnet50_layers)
    for BATCH in [32]
]


def bench_op(
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    # parameters of conv
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    dtype=torch.float32,
    layout="nchw",
    # layout="nhwc",
):
    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn(
        (KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W), dtype=dtype, device="cuda"
    )
    if layout == "nchw":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (
        IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]
    ) // stride[0]
    OUT_W = (
        IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]
    ) // stride[1]

    tflops = (
        lambda ms: 2.0
        * BATCH
        * OUT_H
        * OUT_W
        * IN_C
        * KERNEL_H
        * KERNEL_W
        * KERNEL_N
        / ms
        * 1e-9
    )

    def fn_cublas():
        return torch.conv2d(x, w, None, stride, padding, dilation, groups)

    def fn_triton():
        return conv_analytic(
            x, w, None, stride, padding, dilation, False, (0, 0), groups
        )

    correct = fn_cublas()
    actual = fn_triton()
    torch.testing.assert_close(correct, actual, rtol=1e-3, atol=1e-3)

    ms, min_ms, max_ms = do_bench(fn_cublas)
    print("cublas", tflops(ms))
    ms, min_ms, max_ms = do_bench(fn_triton)
    print("triton", tflops(ms))


bench_op(
    **{
        "BATCH": 32,
        "IN_H": 56,
        "IN_W": 56,
        "IN_C": 64,
        "KERNEL_N": 128,
        "KERNEL_H": 3,
        "KERNEL_W": 3,
        "stride": (1, 1),
        "padding": (0, 0),
        "groups": 8,
    }
)


for conv in conv_confs:
    if conv["KERNEL_H"] == 1:
        continue
    print(conv)
    bench_op(**conv)
