# flake8: noqa
import model
import torch

import torch._dynamo
import torch._inductor.config
import triton
from prettytable import PrettyTable

# torch._inductor.config.debug = True
torch._inductor.config.triton.convolution = "triton"
torch._inductor.config.triton.dense_indexing = True
torch.manual_seed(0)
useCudaGraph = True


class Func(object):
    # conv
    @torch._dynamo.optimize("inductor")
    def conv_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return y

    # conv
    def conv(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return y

    # conv+bias
    @torch._dynamo.optimize("inductor")
    def conv_add_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return y

    # conv+bias
    def conv_add(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return y

    # relu(conv)
    @torch._dynamo.optimize("inductor")
    def conv_relu_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv)
    def conv_relu(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv+bias)
    @torch._dynamo.optimize("inductor")
    def conv_add_relu_torchinductor(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return torch.relu(y)

    # relu(conv+bias)
    def conv_add_relu(x, w, bias, stride, padding, dilation, groups):
        y = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
        return torch.relu(y)

    # bn(conv)
    @torch._dynamo.optimize("inductor")
    def conv_bn_torchinductor(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        groups,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
    ):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y,
            weight=bn_weight,
            bias=bn_bias,
            running_mean=running_mean,
            running_var=running_var,
            training=False,
            momentum=1,
            eps=1e-5,
            cudnn_enabled=True,
        )
        return y

    # bn(conv)
    def conv_bn(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        groups,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
    ):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y,
            weight=bn_weight,
            bias=bn_bias,
            running_mean=running_mean,
            running_var=running_var,
            training=False,
            momentum=1,
            eps=1e-5,
            cudnn_enabled=True,
        )
        return y

    # relu(bn(conv))
    @torch._dynamo.optimize("inductor")
    def conv_bn_relu_torchinductor(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        groups,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
    ):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y,
            weight=bn_weight,
            bias=bn_bias,
            running_mean=running_mean,
            running_var=running_var,
            training=False,
            momentum=1,
            eps=1e-5,
            cudnn_enabled=True,
        )
        return torch.relu(y)

    # relu(bn(conv))
    def conv_bn_relu(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        groups,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
    ):
        y = torch.conv2d(x, w, None, stride, padding, dilation, groups)
        y = torch.batch_norm(
            y,
            weight=bn_weight,
            bias=bn_bias,
            running_mean=running_mean,
            running_var=running_var,
            training=False,
            momentum=1,
            eps=1e-5,
            cudnn_enabled=True,
        )
        return torch.relu(y)


def cuda_graph(fn, x, w, bias):
    new_x = x.clone()
    new_w = w.clone()
    if bias is not None:
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
        if bias is not None:
            bias.copy_(new_bias)
        return g.replay()

    return fn


def bench(layer_params, layer_id, p, fusion_types=[""]):
    BATCH = 32
    IN_H, IN_W, IN_C, KERNEL_H, KERNEL_W, KERNEL_N, stride, padding = layer_params
    dilation, groups = (1, 1), 1
    dtype = torch.float32

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

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device="cuda")
    w = torch.randn(
        (KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W), dtype=dtype, device="cuda"
    )

    row = [layer_id]
    for fusion_type in fusion_types:

        if fusion_type == "":
            conv_torchinductor = getattr(Func, "conv_torchinductor")
            conv = getattr(Func, "conv")
        else:
            conv_torchinductor = getattr(Func, f"conv_{fusion_type}_torchinductor")
            conv = getattr(Func, f"conv_{fusion_type}")

        if "add" in fusion_type:
            bias = torch.randn((KERNEL_N,), dtype=dtype, device="cuda")
        else:
            bias = None

        args = (x, w, bias, stride, padding, dilation, groups)

        if "bn" in fusion_type:
            running_mean = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
            running_var = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
            bn_weight = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
            bn_bias = torch.randn((KERNEL_N), dtype=dtype, device="cuda")
            args += (
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
            )

        def fn_conv():
            return conv(*args)

        def fn_conv_torchinductor():
            return conv_torchinductor(*args)

        if useCudaGraph:
            fn_conv = cuda_graph(fn_conv, x, w, bias)

        torch_conv_ms, _, _ = triton.testing.do_bench(fn_conv)
        triton_conv_ms, _, _ = triton.testing.do_bench(fn_conv_torchinductor)
        row.extend([tflops(torch_conv_ms), tflops(triton_conv_ms)])

    p.add_row(row)


fusion_types = ["", "add", "relu", "add_relu", "bn", "bn_relu"]
p = PrettyTable()
field_names = ["layer"]
for fusion_type in fusion_types:
    if fusion_type == "":
        field_names.append("torch conv")
        field_names.append("triton conv")
    else:
        field_names.append(f"torch conv+{fusion_type}")
        field_names.append(f"triton conv+{fusion_type}")

p.field_names = field_names
p.float_format = ".3"
for id, layer in enumerate(model.resnet50_layers):
    bench(layer, id, p, fusion_types)

print(p)
