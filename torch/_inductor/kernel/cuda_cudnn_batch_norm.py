# mypy: allow-untyped-defs
import math

import torch

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..utils import cudnn_batch_norm_use_single_read
from ..virtualized import V
from .cuda_batch_norm import cuda_bn_statistics_grid
from .mm_common import load_kernel_template


cudnn_bn_statistics_template = TritonTemplate(
    name="cudnn_bn_statistics",
    grid=cuda_bn_statistics_grid,
    source=load_kernel_template("triton_cudnn_bn_statistics"),
)


@SymbolicGridFn
def cudnn_bn_nhwc_grid(*args):
    meta = args[-1]
    tiles = (meta["C"] + meta["TILE"] - 1) // meta["TILE"]
    return (tiles, 1, 1) if meta["FINAL"] else (meta["PARTS"], tiles, 1)


cudnn_bn_nhwc_template = TritonTemplate(
    name="cudnn_bn_nhwc",
    grid=cudnn_bn_nhwc_grid,
    source=load_kernel_template("triton_cudnn_bn_nhwc"),
)


def cudnn_bn_statistics(input, running_mean, running_var, momentum, eps):
    r"""Generate cuDNN training statistics and functional running-stat updates."""
    from torch._prims_common import suggest_memory_format

    device = input.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    node = V.graph.current_node.args[0]
    if not isinstance(node, torch.fx.Node):
        raise AssertionError(f"Expected torch.fx.Node, got {type(node)!r}")
    value = node.meta["val"]
    channels_last = input.get_dtype() != torch.float64 and suggest_memory_format(
        value
    ) in (
        torch.channels_last,
        torch.channels_last_3d,
    )
    input = (
        L.constrain_to_fake_tensor(input, value)
        if channels_last
        else ir.ExternKernel.require_contiguous(input)
    )
    shape = [V.graph.sizevars.guard_int(size) for size in input.get_size()]
    n, c, spatial = shape[0], shape[1], math.prod(shape[2:])
    dtype = torch.float64 if input.get_dtype() == torch.float64 else torch.float32
    vec = 2 if input.get_dtype() == torch.float16 and spatial % 2 == 0 else 1
    welford = not cudnn_batch_norm_use_single_read(
        input.get_dtype(), n, spatial, device
    )
    bw = 512 if n == 1 or spatial >= 4096 else 128 if spatial >= 1024 else 32
    if running_mean is None:
        running_mean = L.full([c], 0, dtype=dtype, device=device)
    if running_var is None:
        running_var = L.full([c], 1, dtype=dtype, device=device)
    running_mean = ir.ExternKernel.require_contiguous(running_mean)
    running_var = ir.ExternKernel.require_contiguous(running_var)
    for value in (input, running_mean, running_var):
        value.realize()
    if channels_last:
        vec = 4 if c % 4 == 0 else 1
        tile = (4 if c < 4 else 16) * vec
        samples = 5 if input.get_dtype() == torch.float16 else 2
        span = samples * 512 * vec // tile
        parts = min(256, (n * spatial + span - 1) // span)
        options = dict(
            RANK=len(shape), C=c, TILE=tile, VEC=vec, SAMPLES=samples, PARTS=parts
        )
        partial = cudnn_bn_nhwc_template.generate(
            input_nodes=(input,),
            layout=ir.FixedLayout(device, dtype, [3, parts, c]),
            num_stages=1,
            num_warps=16,
            triton_meta={"enable_fp_fusion": False},
            FINAL=False,
            **options,
        )
        choice = cudnn_bn_nhwc_template.generate(
            input_nodes=(input, partial.output_node(), running_mean, running_var),
            layout=ir.FixedLayout(device, dtype, [4, c]),
            num_stages=1,
            num_warps=16,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            EPS=eps,
            MOMENTUM=momentum,
            **options,
        )
        output = choice.output_node()
        return tuple(L.clone(L.select(output, 0, row)) for row in range(4))
    choice = cudnn_bn_statistics_template.generate(
        input_nodes=(input, running_mean, running_var),
        layout=ir.FixedLayout(device, dtype, [4, c]),
        num_stages=1,
        num_warps=16,
        triton_meta={"enable_fp_fusion": False},
        RANK=len(shape),
        C=c,
        WELFORD=welford,
        VEC=vec,
        BW=bw,
        EPS=eps,
        MOMENTUM=momentum,
    )
    output = choice.output_node()
    return tuple(L.clone(L.select(output, 0, row)) for row in range(4))


@SymbolicGridFn
def cudnn_bn_backward_grid(*args):
    meta = args[-1]
    if meta["CHANNELS_LAST"]:
        tiles = (meta["C"] + meta["TILE"] - 1) // meta["TILE"]
        return (tiles, 1, 1) if meta["FINAL"] else (meta["PARTS"], tiles, 1)
    return (meta["C"], 1, 1)


cudnn_bn_backward_statistics_template = TritonTemplate(
    name="cudnn_bn_backward_statistics",
    grid=cudnn_bn_backward_grid,
    source=load_kernel_template("triton_cudnn_bn_backward_statistics"),
)


@SymbolicGridFn
def cudnn_bn_backward_input_grid(*args):
    meta = args[-1]
    return ((math.prod(args[:-1]) + meta["BLOCK"] - 1) // meta["BLOCK"], 1, 1)


cudnn_bn_backward_input_template = TritonTemplate(
    name="cudnn_bn_backward_input",
    grid=cudnn_bn_backward_input_grid,
    source=load_kernel_template("triton_cudnn_bn_backward_input"),
)


def cudnn_bn_backward(grad, input, weight, save_mean, save_invstd):
    r"""Generate cuDNN gradients using its reduction and affine arithmetic orders."""
    from torch._prims_common import (
        make_channels_last_strides_for,
        make_contiguous_strides_for,
        suggest_memory_format,
    )

    device = input.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    node = V.graph.current_node.args[1]
    if not isinstance(node, torch.fx.Node):
        raise AssertionError(f"Expected torch.fx.Node, got {type(node)!r}")
    value = node.meta["val"]
    memory_format = suggest_memory_format(value)
    channels_last = input.get_dtype() != torch.float64 and memory_format in (
        torch.channels_last,
        torch.channels_last_3d,
    )
    input = (
        L.constrain_to_fake_tensor(input, value)
        if channels_last
        else ir.ExternKernel.require_contiguous(input)
    )
    grad = (
        L.constrain_to_fake_tensor(grad, value)
        if channels_last
        else ir.ExternKernel.require_contiguous(grad)
    )
    weight = ir.ExternKernel.require_contiguous(weight)
    save_mean = ir.ExternKernel.require_contiguous(save_mean)
    save_invstd = ir.ExternKernel.require_contiguous(save_invstd)
    shape = [V.graph.sizevars.guard_int(size) for size in input.get_size()]
    n, c, spatial = shape[0], shape[1], math.prod(shape[2:])
    dtype = torch.float64 if input.get_dtype() == torch.float64 else torch.float32
    vec = 2 if input.get_dtype() == torch.float16 and spatial % 2 == 0 else 1
    single = cudnn_batch_norm_use_single_read(
        input.get_dtype(), n, spatial, device, backward=True
    )
    bw = 512 if n == 1 or spatial >= 4096 else 128 if spatial >= 1024 else 32
    options = dict(
        RANK=len(shape), C=c, CHANNELS_LAST=channels_last, SINGLE=single, VEC=vec, BW=bw
    )
    for tensor in (input, grad, weight, save_mean, save_invstd):
        tensor.realize()
    if channels_last:
        vec = 4 if c % 4 == 0 else 1
        tile = (4 if c < 4 else 16) * vec
        rows = 512 * vec // tile
        parts = min(256, (n * spatial + 3 * rows - 1) // (3 * rows))
        streamed = (n * spatial + parts * rows - 1) // (parts * rows) > 6
        if streamed:
            parts = 2 * torch.cuda.get_device_properties(device).multi_processor_count
        groups = (
            max(1, (n * spatial + parts * 3 * rows - 1) // (parts * 3 * rows) - 1)
            if streamed
            else 1
        )
        extra = (
            3
            if streamed
            else max(0, (n * spatial + parts * rows - 1) // (parts * rows) - 3)
        )
        options.update(
            TILE=tile,
            VEC=vec,
            PARTS=parts,
            GROUPS=groups,
            EXTRA=extra,
            STREAMED=streamed,
        )
        partial = cudnn_bn_backward_statistics_template.generate(
            input_nodes=(input, grad, save_mean),
            layout=ir.FixedLayout(device, dtype, [2, parts, c]),
            num_stages=1,
            num_warps=16,
            triton_meta={"enable_fp_fusion": False},
            FINAL=False,
            **options,
        )
        inputs = (partial.output_node(),)
    else:
        inputs = (input, grad, save_mean)
    final = cudnn_bn_backward_statistics_template.generate(
        input_nodes=inputs,
        layout=ir.FixedLayout(device, dtype, [2, c]),
        num_stages=1,
        num_warps=16,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    statistics = final.output_node()
    strides = (
        make_channels_last_strides_for(input.get_size())
        if memory_format in (torch.channels_last, torch.channels_last_3d)
        else make_contiguous_strides_for(shape)
    )
    output = cudnn_bn_backward_input_template.generate(
        input_nodes=(input, grad, weight, save_mean, save_invstd, statistics),
        layout=ir.FixedLayout(device, input.get_dtype(), shape, strides),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        BLOCK=512,
        ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
        OUTPUT_INDICES=("n", "channel", *(f"i{dim}" for dim in range(2, len(shape)))),
        **options,
    )
    grad_weight = L.mul(L.select(statistics, 0, 1), save_invstd)
    grad_bias = L.clone(L.select(statistics, 0, 0))
    return output.output_node(), grad_weight, grad_bias
