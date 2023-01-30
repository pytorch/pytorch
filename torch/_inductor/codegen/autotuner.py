import builtins

import torch
from torch._dynamo.testing import rand_strided

from .. import config, triton_ops
from ..virtualized import V

aten = torch.ops.aten


def str2func(str):
    module, *name = str.split(".")
    if module == "aten":
        runnable = aten
    elif module == "triton_ops":
        runnable = triton_ops
    elif module == "torch":
        runnable = torch
    else:
        raise Exception(f"{str} could not be called")

    for n in name:
        runnable = getattr(runnable, n)
    return runnable


class Autotuner:
    def __init__(self):
        self.cache = dict()

    def _bench(self, kernel, *args, **kwargs):
        def kernel_call():
            kernel(*args, **kwargs)

        from triton.testing import do_bench

        return do_bench(kernel_call)


autotune = Autotuner()


def tuned_conv(
    x_shape,
    w_shape,
    x_stride,
    w_stride,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    device,
    dtype,
    adjust_triton=0.95,
):
    """
    Return the best kernel name given inputs and layer parameters;
    Considering potential pointwise fusion of conv, we could adjust triton timing
    by multiplying adjust_triton (default=0.95)
    """

    sizevars = V.graph.sizevars
    x_shape = [sizevars.size_hint(s) for s in x_shape]
    w_shape = [sizevars.size_hint(s) for s in w_shape]
    x_stride = [sizevars.size_hint(s) for s in x_stride]
    w_stride = [sizevars.size_hint(s) for s in w_stride]
    x = rand_strided(x_shape, x_stride, device=device, dtype=dtype)
    w = rand_strided(w_shape, w_stride, device=device, dtype=dtype)
    # the identifiable args for the layers
    id_args = [
        *x_shape,
        *w_shape,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        # *x_stride,
        # *w_stride,
    ]
    use_cuda = x.is_cuda

    # gen_key
    key = tuple(id_args)
    key = ("conv",) + key

    # candidate kernels
    kernels = ["aten.convolution"]
    if use_cuda:
        kernels += ["triton_ops.conv"]

    # filter kernels that args/kwargs does not meet requirements
    remove_kernels = []
    if groups > 1 or transposed:
        remove_kernels += ["triton_ops.conv"]
    kernels = [k for k in kernels if k not in remove_kernels]

    # if only one choice, return that kernel
    if len(kernels) == 1:
        kernel = kernels[0]
        # return kernel(
        #     x, w, stride, padding, dilation, transposed, output_padding, groups
        # )
        return kernel
    timings = {}
    if key not in autotune.cache:
        for kernel in kernels:
            runnable_kernel = str2func(kernel)
            if "triton_ops" in kernel:
                # because we use nhwc layout by default for triton conv
                x = x.to(memory_format=torch.channels_last)
            run_args = (
                x,
                w,
                None,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            timing, _, _ = autotune._bench(runnable_kernel, *run_args)
            if "triton_ops" in kernel:
                timing = timing * adjust_triton
            timings[kernel] = timing
        autotune.cache[key] = builtins.min(timings, key=timings.get)
        if config.debug:
            print("for key = ", key)
            print("timing", timings)
            print("best_kernel", autotune.cache[key])
    best_kernel = autotune.cache[key]
    # if best_kernel == "triton_ops.conv":
    #     print(key, best_kernel)
    return best_kernel


def tuned_conv_layout(
    kernel,
    x_shape,
    w_shape,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    device,
    dtype,
):
    sizevars = V.graph.sizevars
    x_shape = [sizevars.size_hint(s) for s in x_shape]
    w_shape = [sizevars.size_hint(s) for s in w_shape]
    x = torch.randn(x_shape, device=device, dtype=dtype)
    w = torch.randn(w_shape, device=device, dtype=dtype)
    id_args = [
        *x_shape,
        *w_shape,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ]

    # gen_key
    key = tuple(id_args)
    key = ("conv_layout",) + key
    runnable_kernel = str2func(kernel)

    timings = {}
    if key not in autotune.cache:
        for memory_format in ["torch.contiguous_format", "torch.channels_last"]:
            x = x.to(memory_format=str2func(memory_format))
            run_args = (
                x,
                w,
                None,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            timing, _, _ = autotune._bench(runnable_kernel, *run_args)
            timings[memory_format] = timing
        autotune.cache[key] = builtins.min(timings, key=timings.get)
        if config.debug:
            print("for key = ", key)
            print("timing", timings)
            print("best_layout", autotune.cache[key])
    best_layout = autotune.cache[key]
    return best_layout
