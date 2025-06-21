# mypy: allow-untyped-defs
import functools
from typing import Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args

from . import config, ir
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_grouped_gemm_template import CppGroupedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
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
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import ops, OpsValue, V


def create_int8_compensation(
    W_tensor: torch.Tensor,
    packed_weight: ir.TensorBox,
    x_scale: ir.TensorBox,
    x_zp: ir.TensorBox,
    w_scale: ir.TensorBox,
) -> tuple[bool, ir.TensorBox, Optional[ir.TensorBox]]:
    use_int8_fast_compensation_path = False
    weight_compens = None
    x_w_scale = None
    if all(
        isinstance(item, ir.TensorBox)
        and item.get_name() in V.graph.constants
        and hasattr(item.data, "data")
        and isinstance(item.data.data, ir.ConstantBuffer)
        for item in [x_scale, x_zp, w_scale]
    ):
        use_int8_fast_compensation_path = True
        x_w_scale_tensor = (
            V.graph.constants[x_scale.get_name()]
            * V.graph.constants[w_scale.get_name()]
        )
        x_w_scale = V.graph.add_tensor_constant(
            x_w_scale_tensor,
            name=packed_weight.get_name() + "_x_w_compens",
        )
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        x_zp_tensor = V.graph.constants[x_zp.get_name()]
        weight_compens_tensor = weight_compens_tensor * x_w_scale_tensor * x_zp_tensor
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    else:
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    return (
        use_int8_fast_compensation_path,
        weight_compens,
        x_w_scale,
    )


def codegen_int8_gemm_template_compensation(
    use_int8_fast_compensation_path: bool,
    input: OpsValue,
    _weight_compo: OpsValue,
    _x_scale: Optional[OpsValue],
    _x_zp: Optional[OpsValue],
    _w_scale: Optional[OpsValue],
    _x_w_scale: Optional[OpsValue],
) -> OpsValue:
    if use_int8_fast_compensation_path:
        temp = ops.sub(
            ops.mul(
                input,
                _x_w_scale,
            ),
            _weight_compo,
        )
    else:
        temp = ops.mul(
            ops.mul(
                input,
                _x_scale,
            ),
            _w_scale,
        )
        # NOTE: We will apply compensation even if the x_zp is 0 for int8 quantization.
        # That's because when torch.compile is invoked for dynamic quantization,
        # x might coincidentally have such values that x_zp might be zero despite
        # asymmetric quantization.
        # Besides, if x_zp is dummy for int8 x, or if x is statically quantized,
        # we'd still perform that redundant compute to avoid making the code messy
        # because we discovered that redundant computation of compensation did not
        # lead to performance degradation with the input shapes tested.
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
    return temp


def grouped_gemm_lowering(
    x: TensorBox,
    w: list[TensorBox],
    b: list[TensorBox],
    attr=None,
    scalars=None,
    algorithm=None,
    layout=None,
):
    x_size = x.get_size()
    if len(x_size) > 2:
        # GEMM template needs 2D input, normalize input shape here
        x = view(x, [-1, x_size[-1]])
    num_gemm = len(w)

    assert config.max_autotune or config.max_autotune_gemm
    b = [bias if bias is None else ir.ExternKernel.realize_input(bias) for bias in b]

    choices: list[ChoiceCaller] = []
    *_, layout, x, _ = mm_args(x, permute(w[0], [1, 0]), layout=layout)

    kwargs = dict(
        has_bias=[bias is not None for bias in b],
        trans_w=True,
        epilogue_creator=None,
        act_mapping=dict.fromkeys(range(num_gemm), x),
    )

    input_nodes = [x, *w]
    input_nodes.extend([bias for bias in b if bias is not None])

    CppGroupedGemmTemplate.add_choices(
        choices,
        layout,
        input_nodes,
        **kwargs,  # type: ignore[arg-type]
    )

    assert len(choices) != 0
    result = autotune_select_algorithm(
        "grouped_gemm",
        choices,
        input_nodes,
        layout,
    )
    template_buf = result.data.data
    return_bufs = [
        ir.MultiOutput(layout, template_buf, [(list, gemm_idx)])
        for gemm_idx in range(num_gemm)
    ]
    template_buf.layout = ir.MultiOutputLayout(device=input_nodes[0].get_device())
    template_buf.outputs = return_bufs
    return_tensors = [
        ir.TensorBox.create(return_bufs[gemm_idx]) for gemm_idx in range(num_gemm)
    ]
    if len(x_size) > 2:
        for gemm_idx in range(num_gemm):
            return_tensors[gemm_idx] = view(
                return_tensors[gemm_idx],
                (*x_size[:-1], return_tensors[gemm_idx].get_size()[-1]),
            )
    return return_tensors


grouped_gemm_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        from . import mkldnn_ir

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
        aten_mkldnn_qlinear_binary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise.binary,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwiseBinaryPT2E.create,
        )
        cpu_needs_realized_inputs = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
            torch.ops.onednn.qconv_pointwise,
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
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                if use_cpp_gemm_template(layout, x, transposed_w):

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
                    CppGemmTemplate.add_choices(
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
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w, y = mm_args(
                    x, transposed_w, y, layout=layout
                )
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(buf, attr, other=y)

                    kwargs = dict(
                        has_bias=b is not None,
                        trans_w=True,
                        epilogue_creator=epilogue_creator,
                    )
                    kwargs["input_indices"] = [0, 2, 1] if b is None else [3, 0, 2, 1]
                    CppGemmTemplate.add_choices(
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
            batch_sizes: list[int],
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

        @register_lowering(torch.ops.onednn.qconv_pointwise, type_promotion_kind=None)
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
            # To align with qlinear where x_scale and x_zp are converted to Tensor
            assert type(x_scale) == float
            x_scale = V.graph.add_tensor_constant(
                torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
            )
            assert type(x_zp) == int
            x_zp = V.graph.add_tensor_constant(
                torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
            )

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
        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qconvolution_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            accum: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithmm,
        ):
            # To align with qlinear where x_scale and x_zp are converted to Tensor
            assert type(x_scale) == float
            x_scale = V.graph.add_tensor_constant(
                torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
            )
            assert type(x_zp) == int
            x_zp = V.graph.add_tensor_constant(
                torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
            )

            if (
                binary_attr == "sum"
                and output_dtype in [torch.float32, torch.bfloat16]
                and accum.get_dtype() in [torch.float32, torch.bfloat16]
                and accum.get_dtype() != output_dtype
            ):
                # For int8-mixed-bf16 quantization and inplace add,
                # there is case when accum dtype is float32 but output dtype is bfloat16.
                # Since the accum will be inplaced changed with post op sum,
                # we will do accum dtype conversion here.
                accum = to_dtype(accum, output_dtype)
            return TensorBox.create(
                mkldnn_ir.QConvPointWiseBinaryPT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    accum,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    accum_scale,
                    accum_zp,
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
            assert packed_weight.get_dtype() is torch.int8, (
                "Only int8 weights are supported by oneDNN qlinear."
            )
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
                if all(dim == 1 for dim in x_scale.get_size()):
                    # Corner-case discovered with LLaMA series.
                    # If all outer dims of x_scale are 1, make it a 0D tensor.
                    # Otherwise, epilogue creator will run into indexing issues.
                    x_scale = view(x_scale, [])
                assert len(x_scale.get_size()) in [0, 1], "x_scale must be 0D or 1D"

            if x_zp is None:
                # If x_zp is None, x is int8 quantized per-tensor and its scale is not reshaped,
                # then the codegened code would segfault if we don't create a tensor for x_zp.
                # It's safe to do so since x is a symmetrically quantized int8 tensor.
                # Moreover, oneDNN qlinear API doesn't accept None value for zp
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            if not isinstance(x_zp, ir.TensorBox):
                assert type(x_zp) == int
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp.realize()

            assert x_zp.get_numel() == 1, "x_zp is incompatible with oneDNN qlinear"

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577  # noqa: B950
            if w_zp is None:
                # If w_zp is None, then it's a dummy tensor created to denote the
                # absence of a zero point, and thus w is int8 symmetrically quantized.
                # Moreover, oneDNN qlinear API doesn't accept None value for zp
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )
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
            choices: list[ChoiceCaller] = []

            if config.max_autotune or config.max_autotune_gemm:
                *_, layout, x, packed_weight = mm_args(
                    x, packed_weight, layout=layout, out_dtype=output_dtype
                )

                if (
                    # GEMM template currently only supports symmetrically quantized weights
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )
                ) and use_cpp_gemm_template(layout, x, packed_weight):
                    W_tensor = V.graph.constants[packed_weight.get_name()].to_dense()

                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        assert output_dtype in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]
                        input_loader = input_buffer.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            assert x_w_scale is not None
                            x_w_scale_loader = x_w_scale.make_loader()
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

                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                assert x_w_scale_loader is not None
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Compute s8s8->s32 or u8s8->s32 GEMM & then apply compensation
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
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
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32 & s8s8f32
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
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    assert x.get_dtype() in [torch.uint8, torch.int8]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        input_indices=[0, 3, 1, 2, 4, 5]
                        if bias is None
                        else [6, 0, 3, 1, 2, 4, 5],
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
                3: lambda x: V.graph.constants[x.get_name()],  # packed weight
                4: lambda x: V.graph.constants[x.get_name()],  # weight scale
                5: lambda x: V.graph.constants[x.get_name()],  # weight zp
                6: lambda x: V.graph.constants[x.get_name()],  # bias
            }
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_scale),
                ir.ConstantBuffer,
            ):
                # x is statically quantized
                input_gen_fns[1] = lambda x: V.graph.constants[x.get_name()]
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_zp),
                ir.ConstantBuffer,
            ):
                input_gen_fns[2] = lambda x: V.graph.constants[x.get_name()]

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
            x2: TensorBox,
            bias: TensorBox,
            o_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithmm,
            layout=None,
        ):
            x_size = x.get_size()
            x2_size = x2.get_size()
            assert len(x_size) == len(x2_size)
            if len(x_size) > 2 and binary_attr == "add":
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
                x2 = view(x2, [-1, x2_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                assert type(x_scale) == float
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale.realize()
                if all(dim == 1 for dim in x_scale.get_size()):
                    # Corner-case discovered with LLaMA series.
                    # If all outer dims of x_scale are 1, make it a 0D tensor.
                    # Otherwise, epilogue creator will run into indexing issues.
                    x_scale = view(x_scale, [])
                assert len(x_scale.get_size()) in [0, 1], "x_scale must be 0D or 1D"

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            if not isinstance(x_zp, ir.TensorBox):
                assert type(x_zp) == int
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp.realize()

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577  # noqa: B950
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )
            if binary_attr == "sum":
                if output_dtype in [
                    torch.float32,
                    torch.bfloat16,
                ] and x2.get_dtype() in [torch.float32, torch.bfloat16]:
                    if x2.get_dtype() != output_dtype:
                        # For int8-mixed-bf16 quantization and inplace add,
                        # there is case when accum dtype is float32 but output dtype is bfloat16.
                        # Since the accum will be inplaced changed with post op sum,
                        # we will do accum dtype conversion here.
                        x2 = to_dtype(x2, output_dtype)
                else:
                    assert x2.get_dtype() == output_dtype, (
                        "dtype of accum for qlinear post op sum should be the same as output"
                    )
            x2_dtype = x2.get_dtype()
            bias_dtype = bias.get_dtype() if bias is not None else None
            choices: list[ChoiceCaller] = []
            if (
                config.max_autotune or config.max_autotune_gemm
            ) and binary_attr == "add":  # <TODO> Support inplace sum fusion
                *_, layout, x, packed_weight, x2 = mm_args(
                    x, packed_weight, x2, layout=layout, out_dtype=output_dtype
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
                    and use_cpp_gemm_template(layout, x, packed_weight)
                ):
                    W_tensor = V.graph.constants[packed_weight.get_name()]
                    W_tensor = W_tensor.to_dense()
                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        assert output_dtype in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]

                        input_loader = input_buffer.make_loader()
                        x2_loader = x2.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            assert x_w_scale is not None
                            x_w_scale_loader = x_w_scale.make_loader()
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
                            _x2 = x2_loader(index)
                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            weight_compens_index = (index[-1],)
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            # MicroKernel Output is with int32: cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                assert x_w_scale_loader is not None
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Doing compensation to cvt fp32
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                assert bias_dtype in [torch.float32, torch.bfloat16]
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            # Step 3: Binary add
                            nonlocal x2_dtype
                            assert x2_dtype in [torch.float32, torch.bfloat16]
                            if x2_dtype == torch.bfloat16:
                                _x2 = ops.to_dtype(_x2, torch.float32)
                            temp = ops.add(temp, _x2)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 4: Unary post op if has
                        if unary_attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf,
                                unary_attr,
                                scalars=unary_scalars,
                                algorithm=unary_algorithmm,
                            )

                        # Step 5: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, torch.uint8)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=torch.uint8,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        # Reorder bias and x2
                        input_indices=[0, 3, 1, 2, 4, 5, 6]
                        if bias is None
                        else [7, 0, 3, 1, 2, 4, 5, 6],
                    )

            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    other_scale=x2_scale,
                    other_zp=x2_zp,
                    binary_post_op=binary_attr,
                    binary_alpha=alpha,
                    unary_post_op=unary_attr,
                    unary_post_op_args=unary_scalars,
                    unary_post_op_algorithm=unary_algorithmm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_binary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias),
                        layout,
                        **kwargs,
                    )
                )
            assert packed_weight.get_name() in V.graph.constants
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],
                4: lambda x: V.graph.constants[x.get_name()],
                5: lambda x: V.graph.constants[x.get_name()],
            }
            if bias is not None:
                input_gen_fns[7] = lambda x: V.graph.constants[x.get_name()]  # For bias
            result = autotune_select_algorithm(
                "qlinear_binary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2 and binary_attr == "add":
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

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
                choices: list[ChoiceCaller] = []
                if config.max_autotune or config.max_autotune_gemm:
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_gemm_template(layout, x, transposed_w):
                        CppGemmTemplate.add_choices(
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
