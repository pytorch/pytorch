from typing import List

import torch
import torch.utils._pytree as pytree
from . import ir
from .ir import TensorBox
from .lowering import add, add_needs_realized_inputs, aten, register_lowering, to_dtype


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        cpu_needs_realized_inputs = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
            torch.ops.onednn.qconv2d_pointwise,
        ]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionBinary.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionBinaryInplace.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(
            x: TensorBox, w: TensorBox, b: TensorBox, attr, scalars, algorithm
        ):
            return TensorBox.create(
                ir.LinearUnary.create(x, w, b, attr, scalars, algorithm)
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr):
            return TensorBox.create(ir.LinearBinary.create(x, y, w, b, attr))

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.ConvolutionTransposeUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(
            x: TensorBox,
            w0: TensorBox,
            w1: TensorBox,
            w2: TensorBox,
            w3: TensorBox,
            hx: TensorBox,
            cx: TensorBox,
            reverse: bool,
            batch_sizes: List[int],
            mode: int,
            hidden_size: int,
            num_layers: int,
            has_biases: bool,
            bidirectional: bool,
            batch_first: bool,
            train: bool,
        ):
            return pytree.tree_map(
                TensorBox.create,
                ir.MkldnnRnnLayer.create(
                    x,
                    w0,
                    w1,
                    w2,
                    w3,
                    hx,
                    cx,
                    reverse,
                    batch_sizes,
                    mode,
                    hidden_size,
                    num_layers,
                    has_biases,
                    bidirectional,
                    batch_first,
                    train,
                ),
            )

        @register_lowering(torch.ops.onednn.qconv2d_pointwise, type_promotion_kind=None)
        def qconvolution_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.QConvPointWisePT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary, type_promotion_kind=None
        )
        def qconvolution_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            accum: TensorBox,
            accum_scale,
            accum_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithmm,
        ):
            if (
                binary_attr == "sum"
                and output_dtype in [torch.float32, torch.bfloat16]
                and accum.get_dtype() in [torch.float32, torch.bfloat16]
                and accum.get_dtype() != output_dtype
            ):
                # For int8-mixed-bf16 quantization and inplace add,
                # there is case when accum dtype is float32 but output dtype is bfloat16.
                # Since the accum will be inplaced changed with post op sum,
                # we will do accum dtype convertion here.
                accum = to_dtype(accum, output_dtype)
            return TensorBox.create(
                ir.QConvPointWiseBinaryPT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    accum,
                    accum_scale,
                    accum_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    binary_attr,
                    alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithmm,
                )
            )

        @register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
        def qlinear_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                ir.QLinearPointwisePT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        if torch._C.has_mkl:
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: TensorBox,
                batch_size,
            ):
                result = TensorBox.create(
                    ir.MKLPackedLinear.create(x, packed_w, orig_w, batch_size)
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)
    else:
        pass
