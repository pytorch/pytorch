# mypy: allow-untyped-defs
import functools
from typing import List, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args
from . import ir, mkldnn_ir
from .codegen.cpp_gemm_template import CppPackedGemmTemplate
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_packed_gemm_template, use_max_autotune
from .virtualized import ops, V


def create_epilogue_with_attr(input_buffer, attr, **kwargs):
    input_loader = input_buffer.make_loader()
    dtype = input_buffer.get_dtype()
    if attr == "relu":

        def inner_fn(index):
            input = input_loader(index)
            zero = ops.constant(0, dtype)
            return ops.maximum(input, zero)

    elif attr == "gelu":
        assert "algorithm" in kwargs
        if kwargs["algorithm"] == "none":

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const = ops.constant(0.7071067811865476, torch.float)
                result = input * half * (ops.erf(input * const) + one)
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

        else:
            assert kwargs["algorithm"] == "tanh"

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const1 = ops.constant(0.7978845608028654, torch.float)
                const2 = ops.constant(0.044715, torch.float)
                result = (
                    half
                    * input
                    * (
                        one
                        + ops.tanh(const1 * (input + const2 * input * input * input))
                    )
                )
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

    elif attr == "swish":

        def inner_fn(index):
            input = input_loader(index)
            result = input * ops.sigmoid(input)
            return result

    elif attr == "sigmoid":

        def inner_fn(index):
            return ops.sigmoid(input_loader(index))

    elif attr == "tanh":

        def inner_fn(index):
            return ops.tanh(input_loader(index))

    elif attr == "hardswish" or attr == "hardsigmoid":

        def hardsigmoid_float(input):
            zero = ops.constant(0, torch.float)
            six = ops.constant(6, torch.float)
            three = ops.constant(3, torch.float)
            one_over_six = ops.constant(0.16666666666666666, torch.float)
            max = ops.maximum(input + three, zero)
            min = ops.minimum(max, six)
            return min * one_over_six

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = hardsigmoid_float(input)
            if attr == "hardswish":
                result = input * result
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "leaky_relu":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 1
        negative_slope = kwargs["scalars"][0]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            zero = ops.constant(0, torch.float)
            result = ops.where(
                input > zero, input, input * ops.constant(negative_slope, torch.float)
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "hardtanh":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 2
        min_value = kwargs["scalars"][0]
        max_value = kwargs["scalars"][1]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = ops.minimum(
                ops.maximum(input, ops.constant(min_value, torch.float)),
                ops.constant(max_value, torch.float),
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "add" or attr == "sub":
        assert "other" in kwargs
        other = kwargs["other"]
        other_loader = other.make_loader()

        def inner_fn(index):
            op = getattr(ops, attr)
            return op(input_loader(index), other_loader(index))

    else:
        raise ValueError(f"Unsupported epilogue attribute: {attr}")
    return ir.Pointwise(
        device=input_buffer.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input_buffer.get_size(),
    )


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        aten_mkldnn_linear_unary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearUnary.create,
        )
        aten_mkldnn_linear_binary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise.binary,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearBinary.create,
        )
        aten_mkldnn_qlinear_unary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwisePT2E.create,
        )
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
                mkldnn_ir.ConvolutionUnary.create(
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
                mkldnn_ir.ConvolutionBinary.create(
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
                mkldnn_ir.ConvolutionBinaryInplace.create(
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
            x: TensorBox,
            w: TensorBox,
            b: TensorBox,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)
            choices: List[ChoiceCaller] = []
            if use_max_autotune():
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                if use_cpp_packed_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(
                            buf, attr, scalars=scalars, algorithm=algorithm
                        )

                    kwargs = dict(
                        has_bias=b is not None,
                        trans_w=True,
                        epilogue_creator=None if attr == "none" else epilogue_creator,
                    )
                    if b is not None:
                        kwargs["input_indices"] = [2, 0, 1]  # type: ignore[assignment]
                    CppPackedGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, w] if b is None else [x, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr, scalars=scalars, algorithm=algorithm)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_unary.bind(
                        [x, w] if b is None else [x, w, b],
                        layout,
                        **kwargs,
                    )
                )
            assert w.get_name() in V.graph.constants
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
            }
            result = autotune_select_algorithm(
                "linear_unary",
                choices,
                [x, w] if b is None else [x, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(
            x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr, layout=None
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            y_size = y.get_size()
            if len(y_size) > 2:
                y = view(y, [-1, y_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)
            choices: List[ChoiceCaller] = []
            if use_max_autotune():
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w, y = mm_args(
                    x, transposed_w, y, layout=layout
                )
                if use_cpp_packed_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(buf, attr, other=y)

                    kwargs = dict(
                        has_bias=b is not None,
                        trans_w=True,
                        epilogue_creator=epilogue_creator,
                    )
                    kwargs["input_indices"] = [0, 2, 1] if b is None else [3, 0, 2, 1]
                    CppPackedGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, y, w] if b is None else [x, y, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_binary.bind(
                        [x, y, w] if b is None else [x, y, w, b],
                        layout,
                        **kwargs,
                    )
                )
            assert w.get_name() in V.graph.constants
            input_gen_fns = {
                2: lambda x: V.graph.constants[x.get_name()],
            }
            result = autotune_select_algorithm(
                "linear_binary",
                choices,
                [x, y, w] if b is None else [x, y, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

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
                mkldnn_ir.ConvolutionTransposeUnary.create(
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
                mkldnn_ir.MkldnnRnnLayer.create(
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
                mkldnn_ir.QConvPointWisePT2E.create(
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
                mkldnn_ir.QConvPointWiseBinaryPT2E.create(
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
            o_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                assert type(x_scale) == float
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale.realize()
            if not isinstance(x_zp, ir.TensorBox):
                assert type(x_zp) == int
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp.realize()

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to https://github.com/pytorch/pytorch/blob
            # /f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                # W_zp might be a ConstantBuffer with int64, convert it to int32
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )

            bias_dtype = None if bias is None else bias.get_dtype()

            choices: List[ChoiceCaller] = []
            if use_max_autotune():
                *_, layout, x, packed_weight = mm_args(
                    x, packed_weight, layout=layout, out_dtype=output_dtype
                )
                if (
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(x_zp),
                        ir.ConstantBuffer,
                    )
                    and len(x_zp.get_layout().size) == 0  # Per tensor quant of act
                    and isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )  # We only compensate MatrixB and assume B_zp is 0 to avoid the compensation of MatrixA
                    and use_cpp_packed_gemm_template(layout, x, packed_weight)
                ):
                    W_tensor = V.graph.constants[packed_weight.get_name()].to_dense()
                    weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
                    weight_compens = V.graph.add_tensor_constant(
                        weight_compens_tensor,
                        name=packed_weight.get_name() + "_BMatrixCompens",
                    )

                    def epilogue_creator(input_buffer):
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        assert output_dtype in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                        ]
                        input_loader = input_buffer.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_scale_loader = x_scale.make_loader()
                        w_scale_loader = w_scale.make_loader()
                        x_zp_loader = x_zp.make_loader()
                        nonlocal bias
                        bias_loader = None
                        if bias is not None:
                            bias_loader = bias.make_loader()

                        def inner_fn(index):
                            nonlocal bias
                            input = input_loader(index)
                            # MicroKernel Output is with int32
                            # cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            weight_compens_index = (index[-1],)
                            _x_scale = x_scale_loader(())
                            _x_zp = x_zp_loader(())
                            _w_scale = w_scale_loader(weight_compens_index)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            # Step 1: Doing compensation to cvt fp32
                            temp = ops.mul(
                                ops.mul(
                                    input,
                                    _x_scale,
                                ),
                                _w_scale,
                            )
                            temp = ops.sub(
                                temp,
                                ops.mul(
                                    ops.mul(
                                        ops.mul(
                                            _x_scale,
                                            _w_scale,
                                        ),
                                        _x_zp,
                                    ),
                                    _weight_compo,
                                ),
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                assert bias_dtype in [torch.float32, torch.bfloat16]
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 3: Doing the unary post op fusion
                        if attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf, attr, scalars=scalars, algorithm=algorithm
                            )

                        # Step 4: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype == torch.uint8:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                qmin, qmax = _create_constants(
                                    0, 255, dtype=torch.float32
                                )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, torch.uint8)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device(),
                                dtype=output_dtype,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    assert x.get_dtype() == torch.uint8
                    CppPackedGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    post_op_name=attr,
                    post_op_args=scalars,
                    post_op_algorithm=algorithm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_unary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias),
                        layout,
                        **kwargs,
                    )
                )
            assert packed_weight.get_name() in V.graph.constants
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],
                4: lambda x: V.graph.constants[x.get_name()],
                5: lambda x: V.graph.constants[x.get_name()],
                6: lambda x: V.graph.constants[x.get_name()],  # For bias
            }
            result = autotune_select_algorithm(
                "qlinear_unary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary, type_promotion_kind=None
        )
        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qlinear_binary(
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
            x2: TensorBox,
            x2_scale,
            x2_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithmm,
        ):
            if binary_attr == "sum":
                if output_dtype in [
                    torch.float32,
                    torch.bfloat16,
                ] and x2.get_dtype() in [torch.float32, torch.bfloat16]:
                    if x2.get_dtype() != output_dtype:
                        # For int8-mixed-bf16 quantization and inplace add,
                        # there is case when accum dtype is float32 but output dtype is bfloat16.
                        # Since the accum will be inplaced changed with post op sum,
                        # we will do accum dtype convertion here.
                        x2 = to_dtype(x2, output_dtype)
                else:
                    assert (
                        x2.get_dtype() == output_dtype
                    ), "dtype of accum for qlinear post op sum should be the same as output"
            return TensorBox.create(
                mkldnn_ir.QLinearPointwiseBinaryPT2E.create(
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
                    x2,
                    x2_scale,
                    x2_zp,
                    binary_attr,
                    alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithmm,
                )
            )

        if torch._C.has_mkl:
            aten_mkl_linear = ExternKernelChoice(
                torch.ops.mkl._mkl_linear,
                "mkl::_mkl_linear",
                has_out_variant=False,
                kernel_creator=mkldnn_ir.MKLPackedLinear.create,
            )
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: Optional[TensorBox],
                batch_size,
                *,
                layout=None,
            ):
                choices: List[ChoiceCaller] = []
                if use_max_autotune():
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_packed_gemm_template(layout, x, transposed_w):
                        CppPackedGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, packed_w, orig_w],
                            trans_w=True,
                            input_indices=[0, 2],
                        )

                if len(choices) == 0 or use_aten_gemm_kernels():
                    choices.append(
                        aten_mkl_linear.bind(
                            (x, packed_w, orig_w), layout, B=None, batch_size=batch_size
                        )
                    )

                assert packed_w.get_name() in V.graph.constants
                assert orig_w.get_name() in V.graph.constants
                # packed_w is a mkldnn tensor which we can't generate directly
                # so we use the weights from the original tensor in autotune.
                input_gen_fns = {
                    1: lambda x: V.graph.constants[x.get_name()],
                    2: lambda x: V.graph.constants[x.get_name()],
                }
                result: TensorBox = autotune_select_algorithm(
                    "packed_linear",
                    choices,
                    [x, packed_w, orig_w],
                    layout,
                    input_gen_fns=input_gen_fns,
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)
    else:
        pass
