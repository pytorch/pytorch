# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING, TypedDict

import torch
from torch._inductor.codegen.rocm.ck_conv_template import CKGroupedConvFwdTemplate

from .. import config, ir
from ..kernel_inputs import ConvKernelInputs
from ..lowering import (
    add_layout_constraint,
    constrain_to_fx_strides,
    lowerings as L,
    register_lowering,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
    KernelTemplate,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    is_ones,
    is_zeros,
    pad_listlike,
    sympy_product,
    use_ck_conv_template,
    use_triton_template,
)
from ..virtualized import V


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..ir import TensorBox

log = logging.getLogger(__name__)


aten = torch.ops.aten


@SymbolicGridFn
def conv2d_grid(n, c, h, w, meta, *, cdiv):
    return (
        cdiv(n * h * w, meta["BLOCK_M"]),
        cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


@SymbolicGridFn
def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
    return (
        cdiv(n * d * h * w, meta["BLOCK_M"]),
        cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


LOOP_BODY_2D = """
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
"""

"""
This is a relatively simple conv implementation that can likely be
improved.  Many alternate conv versions can be found here:
https://github.com/pytorch/torchdynamo/pull/971
"""
conv2d_template = TritonTemplate(
    name="convolution2d",
    grid=conv2d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_H = {{size("X", 2)}}
    IN_W = {{size("X", 3)}}
    OUT_C = {{size(None, 1)}}
    OUT_H = {{size(None, 2)}}
    OUT_W = {{size(None, 3)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xh = {{stride("X", 2)}}
    stride_xw = {{stride("X", 3)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wh = {{stride("W", 2)}}
    stride_ww = {{stride("W", 3)}}

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_2D
    + """
{% endfor %}
{% endfor %}
{% else %}
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
        """
    + LOOP_BODY_2D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_h", "idx_w"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
)

LOOP_BODY_3D = """
        idx_x_d = d - PADDING_D + idx_y_d * STRIDE_D
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_d * stride_xd)[:, None]
            + (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_d >= 0)[:, None]
            & (idx_x_d < IN_D)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] +
            (d * stride_wd) + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

conv3d_template = TritonTemplate(
    name="convolution3d",
    grid=conv3d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_D = {{size("X", 2)}}
    IN_H = {{size("X", 3)}}
    IN_W = {{size("X", 4)}}
    OUT_C = {{size(None, 1)}}
    OUT_D = {{size(None, 2)}}
    OUT_H = {{size(None, 3)}}
    OUT_W = {{size(None, 4)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xd = {{stride("X", 2)}}
    stride_xh = {{stride("X", 3)}}
    stride_xw = {{stride("X", 4)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wd = {{stride("W", 2)}}
    stride_wh = {{stride("W", 3)}}
    stride_ww = {{stride("W", 4)}}

    ndhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = ndhw % OUT_W
    ndh = ndhw // OUT_W
    idx_y_h = ndh % OUT_H
    nd = ndh // OUT_H
    idx_y_d = nd % OUT_D
    idx_n = nd // OUT_D
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for d in range(KERNEL_D) %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    d = {{d}}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_3D
    + """
{% endfor %}
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for d in range(KERNEL_D):
    #   for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for dijk in range(KERNEL_D * KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (dijk % BLOCK_K_COUNT) * BLOCK_K
        dij = dijk // BLOCK_K_COUNT
        j = dij % KERNEL_W
        di = dij // KERNEL_W
        i = di % KERNEL_H
        d = di // KERNEL_H
        """
    + LOOP_BODY_3D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_d < OUT_D)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_d = idx_y_d[:, None]
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_d", "idx_h", "idx_w"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
)

aten_convolution = ExternKernelChoice(
    torch.convolution,
    "at::convolution",
    has_out_variant=False,
    op_overload=aten.convolution.default,
)


def conv1x1_via_mm(x, w, *, out):
    w = torch.squeeze(torch.squeeze(w, -1), -1)
    return torch.matmul(
        x.permute(0, 2, 3, 1), w.permute(1, 0), out=out.permute(0, 2, 3, 1)
    )


aten_conv1x1_via_mm = ExternKernelChoice(conv1x1_via_mm, None)


class ConvLayoutParams(TypedDict):
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int


def conv_layout(
    x: TensorBox,
    weight: TensorBox,
    bias: Optional[TensorBox],
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    """Determine output layout for a convolution"""
    with V.graph.fake_mode:
        output = torch.ops.aten.convolution(
            ir.ir_node_to_tensor(x, guard_shape=True),
            ir.ir_node_to_tensor(weight, guard_shape=True),
            ir.ir_node_to_tensor(bias, guard_shape=True),
            V.graph.sizevars.size_hints(stride),  # type: ignore[arg-type]
            V.graph.sizevars.size_hints(padding),  # type: ignore[arg-type]
            V.graph.sizevars.size_hints(dilation),  # type: ignore[arg-type]
            transposed,
            V.graph.sizevars.size_hints(output_padding),  # type: ignore[arg-type]
            groups,
        )
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())  # type: ignore[assignment]

    return ir.FixedLayout(
        x.get_device_or_error(),
        x.get_dtype(),
        sizes,
        stride,
    )


def channels_last_order(rank):
    order = list(reversed(range(rank)))
    order.insert(1, order.pop(-1))
    return order


def convert_1x1_conv_to_mm(x, weight, bias):
    # special case for 1x1 convolution, which is actually just a matmul
    rank = len(weight.get_size())
    for _ in range(rank - 2):
        weight = L[aten.squeeze](weight, dim=-1)
    weight = L[aten.permute](weight, [1, 0])

    x = ir.ExternKernel.require_stride_order(x, channels_last_order(rank))
    x_permute = list(range(rank))
    x_permute.append(x_permute.pop(1))
    x = L[aten.permute](x, x_permute)
    *sizes, in_chan = x.get_size()
    x = L[aten.reshape](x, [sympy_product(sizes), in_chan])
    if bias is None:
        result = L[aten.mm](x, weight)
    else:
        result = L[aten.addmm](bias, x, weight)
    result = L[aten.reshape](result, [*sizes, -1])
    result_permute = list(range(rank))
    result_permute.insert(1, result_permute.pop(-1))
    return L[aten.permute](result, result_permute)


@register_lowering(aten.convolution)
def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: Optional[TensorBox],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
):
    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)
    if not isinstance(groups, int):
        groups = V.graph.sizevars.guard_int(groups)
    assert isinstance(groups, int)

    # Need use hint for triton template since the template does not
    # work with a dynamic shape.
    #
    # No need to guard_int for dilation and output_padding
    # since the template is only used when dilation is 1 and output_padding
    # is 0.
    stride = tuple(V.graph.sizevars.guard_int_seq(stride))
    padding = tuple(V.graph.sizevars.guard_int_seq(padding))

    kwargs: ConvLayoutParams = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    device_type = ir.get_device_type(x)

    if len(x.get_size()) == len(weight.get_size()) - 1:
        # add batch dimension to simplify rest of function
        return L[aten.squeeze](
            convolution(L[aten.expand](x, [1, *x.get_size()]), weight, bias, **kwargs),
            dim=0,
        )

    out_chan, in_chan, *kernel_shape = V.graph.sizevars.guard_int_seq(weight.get_size())

    # Always convert conv1D to 2D for Intel GPU.
    # Only conv2D can be converted to channel last layout,
    # which have much better performance.
    if len(x.get_size()) == 3 and len(kernel_shape) == 1 and device_type == "xpu":
        kwargs.update(
            {
                "stride": (1,) + stride,
                "padding": (0,) + padding,
                "dilation": (1,) + dilation,
                "output_padding": (0,) + output_padding,
            }
        )
        # (N, C, L) -> (N, C, 1, L)
        x = L[aten.unsqueeze](x, dim=2)
        weight = L[aten.unsqueeze](weight, dim=2)

        return L[aten.squeeze](
            convolution(x, weight, bias, **kwargs),
            dim=2,
        )

    ndim = len(kernel_shape)
    stride = pad_listlike(stride, ndim)
    padding = pad_listlike(padding, ndim)
    dilation = pad_listlike(dilation, ndim)
    output_padding = pad_listlike(output_padding, ndim)

    def channels_last_conv():
        if V.graph.layout_opt and ndim == 2:
            return True

        layout = conv_layout(x, weight, None, **kwargs)
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.size_hints(layout.stride)
        )
        return req_stride_order == ir.NHWC_STRIDE_ORDER

    autotuning_gemm = config.max_autotune or config.max_autotune_gemm

    if (
        (config.conv_1x1_as_mm or (autotuning_gemm and channels_last_conv()))
        and is_ones(kernel_shape)
        and is_ones(stride)
        and is_zeros(padding)
        and is_ones(dilation)
        and not transposed
        and is_zeros(output_padding)
        and groups == 1
        and V.graph.sizevars.statically_known_gt(sympy_product(x.get_size()), 0)
    ):
        return convert_1x1_conv_to_mm(x, weight, bias)

    if bias is not None and device_type != "cpu":
        # peel off the bias, cudnn is slower with it
        result = convolution(x, weight, None, **kwargs)
        return L[aten.add](
            result, L[aten.view](bias, [result.get_size()[1]] + ndim * [1])
        )

    x.realize()
    weight.realize()

    # ndim can be 1 for convolution in models such as demucs
    # TODO: check if it's beneficial to convert Conv1d to Conv2d and then
    # apply channels last.
    if V.graph.layout_opt and ndim == 2:
        V.graph.num_channels_last_conv += 1
        x = ir.ExternKernel.require_channels_last(x)  # type: ignore[assignment]
        # TODO maybe we can convert weights to channels last just once before
        # running the model.
        weight = ir.ExternKernel.require_channels_last(weight)  # type: ignore[assignment]
        layout = conv_layout(x, weight, None, **kwargs)
    else:
        layout = conv_layout(x, weight, None, **kwargs)
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.size_hints(layout.stride)
        )
        x = ir.ExternKernel.require_stride_order(x, req_stride_order)  # type: ignore[assignment]
        weight = ir.ExternKernel.require_stride_order(weight, req_stride_order)  # type: ignore[assignment]

    # Create ConvKernelInputs for unified template configuration
    # Only include bias in input_nodes when it's not None
    # - For Triton templates: bias is always None here (peeled off earlier), so input_nodes = [x, weight]
    # - For ATEN: input_nodes = [x, weight] when bias is None, [x, weight, bias] when bias is present
    if bias is not None:
        bias.realize()
        bias.freeze_layout()
        V.graph.sizevars.guard_int_seq(bias.get_size())
        input_nodes = [x, weight, bias]
        bias_idx = 2
    else:
        input_nodes = [x, weight]
        bias_idx = None

    kernel_inputs = ConvKernelInputs(
        input_nodes,
        scalars={
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "transposed": transposed,
            "output_padding": output_padding,
            "groups": groups,
        },
        x_idx=0,
        weight_idx=1,
        bias_idx=bias_idx,
    )

    # Build list of templates to try
    templates: list[ExternKernelChoice | KernelTemplate] = []

    if torch._inductor.utils._use_conv_autotune_backend("ATEN"):
        templates.append(aten_convolution)

    if (
        torch._inductor.utils._use_conv_autotune_backend("TRITON")
        and use_triton_template(layout)
        # templates only support these:
        and is_ones(dilation)
        and not transposed
        and is_zeros(output_padding)
        # there are some odd models where this check fails (e.g. shufflenet_v2_x1_0)
        and V.graph.sizevars.statically_known_equals(in_chan * groups, x.get_size()[1])  # type: ignore[arg-type]
    ):
        # 1x1 conv via mm
        if (
            is_ones(kernel_shape)
            and is_ones(stride)
            and is_zeros(padding)
            and groups == 1
        ):
            templates.append(aten_conv1x1_via_mm)

        # Add appropriate template based on ndim
        if ndim == 2:
            templates.append(conv2d_template)
        elif ndim == 3:
            templates.append(conv3d_template)

    # Initialize choices list and extend with template configs
    choices: list[ChoiceCaller] = []
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates,
            "convolution",
        )
    )
    if use_ck_conv_template(layout):
        CKGroupedConvFwdTemplate.add_ck_conv_choices(
            choices,
            layout,
            input_nodes=(x, weight) + ((bias,) if bias is not None else tuple()),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_spatial_dimensions=ndim,
        )
    return autotune_select_algorithm(
        "convolution", choices, kernel_inputs.nodes(), layout
    )


@register_lowering(aten._convolution)
def _convolution(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32,
):
    return convolution(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )


def constrain_conv_to_fx_strides(fx_node, *args, **kwargs):
    assert fx_node.target is torch.ops.aten.convolution.default
    if V.graph.layout_opt:
        return args, kwargs
    else:
        return constrain_to_fx_strides(fx_node, *args, **kwargs)


add_layout_constraint(aten.convolution, constrain_conv_to_fx_strides)
