# mypy: allow-untyped-defs
import math

import torch

from .. import ir, lowering as L
from ..select_algorithm import SymbolicGridFn, TritonTemplate
from ..virtualized import V
from .mm_common import load_kernel_template


@SymbolicGridFn
def cuda_bn_statistics_grid(*args):
    return (args[-1]["C"], 1, 1)


cuda_bn_statistics_template = TritonTemplate(
    name="cuda_bn_statistics",
    grid=cuda_bn_statistics_grid,
    source=load_kernel_template("triton_cuda_bn_statistics"),
)


def cuda_bn_statistics(x):
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    x.realize()
    if V.graph.is_negative_buffer(x.get_name()):
        x = L.clone(x)
    x = ir.ExternKernel.require_contiguous(x)
    size = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    n, c, spatial = size[0], size[1], math.prod(size[2:])
    bw = min(512, max(32, 1 << (spatial - 1).bit_length()))
    dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    choice = cuda_bn_statistics_template.generate(
        input_nodes=(x,),
        layout=ir.FixedLayout(device, dtype, [2, c]),
        num_stages=1,
        num_warps=8,
        triton_meta={"enable_fp_fusion": False},
        N=n,
        C=c,
        S=spatial,
        BW=bw,
        BH=512 // bw,
        ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
    )
    output = choice.output_node()
    return L.clone(L.select(output, 0, 0)), L.clone(L.select(output, 0, 1))


@SymbolicGridFn
def cuda_bn_channels_last_grid(*args):
    meta = args[-1]
    return (meta["C"], 1 if meta["FINAL"] else meta["CTAS"], 1)


cuda_bn_channels_last_template = TritonTemplate(
    name="cuda_bn_channels_last",
    grid=cuda_bn_channels_last_grid,
    source=load_kernel_template("triton_cuda_bn_channels_last"),
)


def cuda_bn_channels_last_statistics(x, value):
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    x = L.constrain_to_fake_tensor(x, value)
    x.realize()
    size = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    c = size[1]
    n = math.prod(size) // c
    pow2 = lambda value: 1 << max(0, value.bit_length() - 1)  # noqa: E731
    bw = min(pow2(c), 32)
    bh = min(pow2((n + 15) // 16), 512 // bw)
    if bw * bh != 512:
        bw = min(pow2(c), 512 // bh)
    ctas = min((n + bh * 16 - 1) // (bh * 16), 128)
    if ctas < 8:
        ctas = 1
    dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    options = dict(
        N=n,
        C=c,
        BH=bh,
        CTAS=ctas,
        ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
    )
    if ctas > 1:
        first = cuda_bn_channels_last_template.generate(
            input_nodes=(x,),
            layout=ir.FixedLayout(device, dtype, [3, c, ctas]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=False,
            **options,
        )
        inputs = (x, first.output_node())
    else:
        inputs = (x,)
    final = cuda_bn_channels_last_template.generate(
        input_nodes=inputs,
        layout=ir.FixedLayout(device, dtype, [2, c]),
        num_stages=1,
        num_warps=4,
        triton_meta={"enable_fp_fusion": False},
        FINAL=True,
        **options,
    )
    output = final.output_node()
    return L.clone(L.select(output, 0, 0)), L.clone(L.select(output, 0, 1))


cuda_bn_backward_fused_template = TritonTemplate(
    name="cuda_bn_backward_fused",
    grid=cuda_bn_statistics_grid,
    source=load_kernel_template("triton_cuda_bn_backward_fused"),
)


cuda_bn_backward_channels_last_template = TritonTemplate(
    name="cuda_bn_backward_channels_last",
    grid=cuda_bn_channels_last_grid,
    source=load_kernel_template("triton_cuda_bn_backward_channels_last"),
)


def cuda_bn_backward_fused(
    grad,
    x,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    training,
    eps,
    mask,
    channels_last=False,
    value=None,
    grad_value=None,
    fused=True,
):
    r"""Generate CUDA batch norm backward with native reduction and arithmetic order."""
    from ..virtualized import ops

    for buffer in (x, grad):
        buffer.realize()
    if V.graph.is_negative_buffer(x.get_name()):
        x = L.clone(x)
    if V.graph.is_negative_buffer(grad.get_name()):
        grad = L.clone(grad)
    if channels_last:
        x = L.constrain_to_fake_tensor(x, value)
        grad = L.constrain_to_fake_tensor(grad, grad_value)
    else:
        x = ir.ExternKernel.require_contiguous(x)
        grad = ir.ExternKernel.require_contiguous(grad)
    size = [V.graph.sizevars.guard_int(s) for s in x.get_size()]
    n, c, spatial = size[0], size[1], math.prod(size[2:])
    dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    device = x.get_device()
    if device is None:
        raise AssertionError(f"Expected non-None values, got {device=}")
    if training:
        if save_mean is None or save_invstd is None:
            raise AssertionError(
                f"Expected non-None values, got {save_mean=}, {save_invstd=}"
            )
        mean, invstd = save_mean, save_invstd
    else:
        mean = L.to_dtype(running_mean, dtype)
        variance = L.to_dtype(running_var, dtype)
        epsilon = L.full([], eps, dtype=dtype, device=device)
        invstd = (
            L.rsqrt(L.add(variance, epsilon))
            if not fused
            else L._div_rn(
                L.full([c], 1, dtype=dtype, device=device),
                L.sqrt(L.add(variance, epsilon)),
            )
        )
        if not fused:
            if save_mean is not None and math.prod(save_mean.get_size()) != 0:
                mean = save_mean
            if save_invstd is not None and math.prod(save_invstd.get_size()) != 0:
                invstd = save_invstd
    mean = ir.ExternKernel.require_contiguous(mean)
    if channels_last:
        count = n * spatial
        pow2 = lambda value: 1 << max(0, value.bit_length() - 1)  # noqa: E731
        bw = min(pow2(c), 32)
        bh = min(pow2((count + 15) // 16), 512 // bw)
        if bw * bh != 512:
            bw = min(pow2(c), 512 // bh)
        ctas = min((count + bh * 16 - 1) // (bh * 16), 128)
        if ctas < 8:
            ctas = 1
        options = dict(
            N=count,
            C=c,
            BH=bh,
            CTAS=ctas,
            ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
        )
        inputs = (x, grad, mean)
        if ctas > 1:
            first = cuda_bn_backward_channels_last_template.generate(
                input_nodes=inputs,
                layout=ir.FixedLayout(device, dtype, [2, c, ctas]),
                num_stages=1,
                num_warps=4,
                triton_meta={"enable_fp_fusion": False},
                FINAL=False,
                **options,
            )
            inputs += (first.output_node(),)
        choice = cuda_bn_backward_channels_last_template.generate(
            input_nodes=inputs,
            layout=ir.FixedLayout(device, dtype, [2, c]),
            num_stages=1,
            num_warps=4,
            triton_meta={"enable_fp_fusion": False},
            FINAL=True,
            **options,
        )
    else:
        bw = min(512, max(32, 1 << (spatial - 1).bit_length()))
        bh = 512 // bw
        if not fused:
            bh = min(1 << (n.bit_length() - 1), 16)
            bw = min(bw, 512 // bh)
        choice = cuda_bn_backward_fused_template.generate(
            input_nodes=(x, grad, mean),
            layout=ir.FixedLayout(device, dtype, [2, c]),
            num_stages=1,
            num_warps=8,
            triton_meta={"enable_fp_fusion": False},
            N=n,
            C=c,
            S=spatial,
            BW=bw,
            BH=bh,
            ACC_TYPE="tl.float64" if dtype == torch.float64 else "tl.float32",
        )
    result = choice.output_node()
    dy, dot = L.select(result, 0, 0), L.select(result, 0, 1)
    weight_dtype = weight.get_dtype() if weight is not None else x.get_dtype()
    if channels_last and weight is None:
        grad_weight = L.empty([0], dtype=mean.get_dtype(), device=device)
        grad_bias = L.empty([0], dtype=mean.get_dtype(), device=device)
    else:
        grad_weight = (
            L.to_dtype(L.mul(dot, invstd), weight_dtype)
            if channels_last or mask[1]
            else None
        )
        grad_bias = L.to_dtype(dy, weight_dtype) if channels_last or mask[2] else None
    if not mask[0]:
        return None, grad_weight, grad_bias
    shape = [1, c, *([1] * (len(size) - 2))]
    norm = (
        L.full([], 1.0 / (n * spatial), dtype=dtype, device=device)
        if not fused
        else L._div_rn(
            L.full([], 1, dtype=dtype, device=device),
            L.full([], n * spatial, dtype=dtype, device=device),
        )
    )
    weight_value = (
        L.to_dtype(weight, dtype)
        if weight is not None
        else L.full([c], 1, dtype=dtype, device=device)
    )
    grad_scale = L.view(L.mul(invstd, weight_value), shape)
    if training:
        delta = L.sub(L.to_dtype(x, dtype), L.view(mean, shape))
        if not fused:
            proj_scale = L.mul(L.mul(L.mul(invstd, invstd), dot), norm)
            grad_value = L.make_pointwise(
                lambda dy, norm, grad: ops.fma(ops.neg(dy), norm, grad)
            )(
                L.expand(L.view(dy, shape), size),
                L.expand(norm, size),
                L.to_dtype(grad, dtype),
            )
            project = L.make_pointwise(
                lambda delta, scale, grad: ops.fma(ops.neg(delta), scale, grad)
            )(delta, L.view(proj_scale, shape), grad_value)
        else:
            grad_mean = L.view(L.mul(dy, norm), shape)
            proj_scale = L.mul(L.mul(L.mul(dot, norm), invstd), invstd)
            project = L.make_pointwise(
                lambda delta, scale, grad: ops.fma(ops.neg(delta), scale, grad)
            )(delta, L.view(proj_scale, shape), L.to_dtype(grad, dtype))
            project = L.sub(project, grad_mean)
        grad_input = L.mul(project, grad_scale)
    else:
        if not fused:
            grad_input = L.mul(
                L.mul(L.to_dtype(grad, dtype), L.view(weight_value, shape)),
                L.view(invstd, shape),
            )
        else:
            grad_input = L.mul(L.to_dtype(grad, dtype), grad_scale)
    return L.to_dtype(grad_input, x.get_dtype()), grad_weight, grad_bias


def cuda_bn_backward_eval_input(grad, x, weight, running_var, save_invstd, eps):
    size = x.get_size()
    dtype = torch.float64 if x.get_dtype() == torch.float64 else torch.float32
    shape = [1, size[1], *([1] * (len(size) - 2))]
    if save_invstd is not None and math.prod(save_invstd.get_size()) != 0:
        invstd = L.to_dtype(save_invstd, dtype)
    else:
        variance = L.to_dtype(running_var, dtype)
        epsilon = L.full([], eps, dtype=dtype, device=x.get_device())
        invstd = L.rsqrt(L.add(variance, epsilon))
    result = L.to_dtype(grad, dtype)
    if weight is not None:
        result = L.mul(result, L.view(L.to_dtype(weight, dtype), shape))
    result = L.mul(result, L.view(invstd, shape))
    return L.to_dtype(result, x.get_dtype()), None, None
