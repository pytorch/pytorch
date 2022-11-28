import model
import torch

import torch._inductor.triton_ops
import triton

# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
useCudaGraph = False

# conv benchmarks
conv_confs = [
    triton.testing.Benchmark(
        x_names=["layout"],
        x_vals=["nchw", "nhwc"],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        ylabel="TFLOPS",
        plot_name=f"resnet50-conv1x1-{i}-performance",
        args={
            "BATCH": BATCH,
            "IN_H": IN_H,
            "IN_W": IN_W,
            "IN_C": IN_C,
            "KERNEL_N": KERNEL_N,
            "KERNEL_H": KERNEL_H,
            "KERNEL_W": KERNEL_W,
            "stride": stride,
            "padding": padding,
        },
    )
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
    if KERNEL_H == 1 and KERNEL_W == 1
    for BATCH in [32]
]


@triton.testing.perf_report(conv_confs)
def bench_op(
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    # provider
    provider,
    # parameters of conv
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    dtype=torch.float32,
    layout="nhwc",
    warmup=25,
    rep=75,
):

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn(
        (KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W), dtype=dtype, device="cuda"
    )
    bias = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
    if layout == "nhwc":
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

    if provider == "cublas":

        def fn():
            return torch.conv2d(x, w, bias, stride, padding, dilation, groups)

    elif provider == "triton":

        def fn():
            return torch._inductor.triton_ops.conv1x1(
                x, w, bias, stride, padding, dilation, False, (0, 0), groups
            )

    if useCudaGraph:
        # prepare new data
        new_x = x.clone()
        new_w = w.clone()
        new_bias = bias.clone()

        # warmp up for cudagraph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()

        def fn():
            x.copy_(new_x)
            w.copy_(new_w)
            bias.copy_(new_bias)
            return g.replay()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


bench_op.run(print_data=True)
