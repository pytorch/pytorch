# mypy: allow-untyped-defs
from typing import Any, List, Optional, Set

import sympy

import torch
from torch._prims_common import make_channels_last_strides_for

from .ir import (
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    ir_node_to_tensor,
    IRNode,
    is_contiguous_storage_and_layout,
    Layout,
    may_convert_to_optional,
    MultiOutput,
    MultiOutputLayout,
    MutationOutput,
    NoneLayout,
    TensorBox,
)
from .utils import convert_shape_to_inductor, pad_listlike
from .virtualized import V


def _prepare_convolution_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    padding: List[int],
    stride: List[int],
    dilation: List[int],
    groups: int,
    transposed: bool = False,
    output_padding: Optional[List[int]] = None,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU/XPU device since conv post-op fusion kernel is only
    supported on CPU/XPU right now.
    """

    # Port from aten/src/ATen/native/ConvUtils.h: _conv_input_size
    def _conv_input_size(
        output_size, weight_size, padding, output_padding, stride, dilation, groups
    ):
        assert len(output_size) == len(weight_size), "Expect input dim == weight dim"
        dim = len(output_size)
        assert dim > 2, "Expect input dim > 2"

        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (
                (output_size[d] - 1) * stride[d - 2]
                - (padding[d - 2] * 2)
                + kernel
                + output_padding[d - 2]
            )
            input_size.append(input_size_d)
        return list(map(int, input_size))

    # The size of prepacked_weight is the prepacked weight size of deconv:
    #   Groups > 1:  [g*o, i/g, ...]
    #   Groups == 1: [o, i, ...]
    # Returns original weight size in [i, o, ...]
    def _original_deconv_weight_size(
        prepacked_weight,
        groups,
    ):
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        assert dim > 2, "Expect weight dim > 2"
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            for d in range(2, dim):
                weight_size.append(prepacked_weight_size[d])
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size

    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        # TODO <Leslie> cleaned up the fake_tensor trace as Linear implementation
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        dims = len(x_fake.size()) - 2
        assert 0 < len(padding) <= dims
        assert 0 < len(dilation) <= dims
        assert 0 < len(stride) <= dims
        padding = pad_listlike(padding, dims)
        dilation = pad_listlike(dilation, dims)
        stride = pad_listlike(stride, dims)
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            assert 0 < len(output_padding) <= dims
            output_padding = pad_listlike(output_padding, dims)
        assert isinstance(groups, (int, sympy.core.numbers.Integer))
        if transposed:
            # When transposed, the size of the prepacked oneDNN weight is different
            # from the PyTorch weight. We're not able to run aten conv with such
            # size. We infer the output size from the input params here:
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(
                input_size,
                weight_size,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
        else:
            bias_fake = (
                ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            )
            output = torch.ops.aten.convolution(
                x_fake,
                weight_fake,
                bias_fake,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            output_size = output.size()

        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order

    x = cls.require_stride_order(x, req_stride_order)

    # We won't do weight prepack for Conv if dynamic_shapes.
    # In static shape cases, since weight is prepacked, we'll always force output to be channels last in the Conv kernel.
    # In dynamic shape cases, for input with channels = 1, like tensor of size (s0, 1, 28, 28) and stride (784, 784, 28, 1),
    # x = cls.require_stride_order(x, req_stride_order) where req_stride_order is in the channels last order
    # won't change the stride of this tensor since stride for dimensions of size 1 is ignored. While in Conv kernel,
    # this tensor is considered as channels first and the output will be in contiguous format.
    # To align the behavior of the Conv kernel, we set the output_stride in such case to be contiguous instead of channels last.
    dynamic_shapes = not all(isinstance(i, int) for i in (output_size))
    if dynamic_shapes and is_contiguous_storage_and_layout(x):
        output_stride = FlexibleLayout.contiguous_strides(output_size)
    else:
        output_stride = make_channels_last_strides_for(output_size)

    assert x.get_device().type in ["xpu", "xpu"] and weight.get_device().type in ["cpu", "xpu"]
    inputs = [x, weight]

    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        convert_shape_to_inductor(output_size),
        convert_shape_to_inductor(output_stride),
    )
    constant_args = [padding, stride, dilation, groups]
    if transposed:
        constant_args.insert(1, output_padding)

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order


def _prepare_linear_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for linear post-op fusion's create function. The function only supports the CPU device
    since linear post-op fusion kernel is only supported on CPU right now.
    """
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()

    *m, _ = x.get_size()
    # The weight has been transposed during the qlinear weight prepack process.
    # https://github.com/pytorch/pytorch/blob/4979f9c0d72490970e2019bb1d2284f83d93f76b/
    # aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp#L291
    _, oc = weight.get_size()
    output_size = list(m) + [oc]
    req_stride_order = list(reversed(range(len(x.get_size()))))

    x = cls.require_stride_order(x, req_stride_order)
    assert x.get_device().type in ["cpu", "xpu"] and weight.get_device().type in ["cpu", "xpu"]
    assert x.get_device().type == weight.get_device().type
    inputs = [x, weight]

    output_stride = FlexibleLayout.contiguous_strides(output_size)
    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        output_size,
        output_stride,
    )
    constant_args: List[Any] = []

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order


class ConvolutionUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._convolution_pointwise",
            cpp_kernel_name="mkldnn::_convolution_pointwise",
        )
        self.cpp_kernel_key = "convolution_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        attr,
        scalars: Optional[List[Any]],
        algorithm,
    ):
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        return ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._convolution_pointwise.binary",
            cpp_kernel_name="mkldnn::_convolution_pointwise",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        return ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ):
        # Due to constrain of op.call, other (Tensor&) should be at input[0]
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        super().__init__(
            kernel_layout,
            reordered_inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._convolution_pointwise_.binary",
            cpp_kernel_name="mkldnn::_convolution_pointwise_",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary_"
        # TODO: op.call: input[0] should be at::Tensor&
        self.cpp_op_schema = """
            at::Tensor&(
                at::Tensor& other_t,
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""

        self.mutation_outputs = [
            MutationOutput(NoneLayout(inputs[0].get_device()), inputs[0], self),
            MutationOutput(NoneLayout(inputs[1].get_device()), inputs[1], self),
        ]

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            _,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]
        packed = ConvolutionBinaryInplace(
            kernel_layout=NoneLayout(inputs[1].get_device()),  # type: ignore[arg-type]
            inputs=inputs,
            constant_args=constant_args,
        )
        # This op mutates in place which means that the result is not the
        # target but rather the input that is being mutated
        # init reorders the inputs, so inputs[1] becomes packed.inputs[0]
        return packed.inputs[0]


class ConvolutionTransposeUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._convolution_transpose_pointwise",
            cpp_kernel_name="mkldnn::_convolution_transpose_pointwise",
        )
        self.cpp_kernel_key = "convolution_transpose_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef output_padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        output_padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups_: int,
        attr,
        scalars: Optional[List[Any]],
        algorithm,
    ):
        transposed = True
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
        ) = _prepare_convolution_fusion_create(
            cls,
            x,
            weight,
            bias,
            padding_,
            stride_,
            dilation_,
            groups_,
            transposed,
            output_padding_,
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        return ConvolutionTransposeUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class QConvPointWisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = len(inputs) == 5
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.onednn.qconv2d_pointwise",
            cpp_kernel_name="onednn::qconv2d_pointwise",
        )
        self.cpp_kernel_key = "qconv2d_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                at::Tensor act,
                double act_scale,
                int64_t act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                torch::List<int64_t> stride,
                torch::List<int64_t> padding,
                torch::List<int64_t> dilation,
                int64_t groups,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        # Parser the inputs and constant
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())

        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        w_scale, w_zp = args[-2], args[-1]
        (
            stride,
            padding,
            dilation,
            groups,
            x_scale,
            x_zp,
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ) = const_args[-12:]

        codegen_args = (
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
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: float,
        x_zero_point: int,
        qw: "TensorBox",  # qw
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ):
        transposed = False
        output_padding = None
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
            cls,
            qx,
            qw,
            bias,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            output_padding,
        )
        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [w_scale, w_zero_point]
        constant_args = constant_args + [
            x_scale,
            x_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]

        assert output_dtype is not None
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_convolution_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set output_dtype is not None, the output buf should be output_dtype instead of uint8.
            kernel_layout.dtype = output_dtype

        return QConvPointWisePT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class QConvPointWiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        """
        Needs input/weight/output qparams
        if bias is not None
            - inputs = [x, w, b, accum, w_scale, w_zp]
            - const_args = [stride, padding, dilation, groups, x_scale, x_zp, accum_scale, accum_zp, o_scale, o_zp,
            fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, accum, w_scale, w_zp]
            - const_args = const_args is: [bias, stride, padding, dilation, groups, x_scale, x_zp, accum_scale,
            accum_zp, o_scale, o_zp, fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = len(inputs) == 6
        self.idx_for_inplace_sum = 3 if self.has_bias else 2
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.onednn.qconv2d_pointwise.binary",
            cpp_kernel_name="onednn::qconv2d_pointwise",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "qconv2d_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                at::Tensor act,
                double act_scale,
                int64_t act_zero_point,
                at::Tensor accum,
                double accum_scale,
                int64_t accum_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                torch::List<int64_t> stride,
                torch::List<int64_t> padding,
                torch::List<int64_t> dilation,
                int64_t groups,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        # Parser the inputs and constant
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())

        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        accum, w_scale, w_zp = args[-3], args[-2], args[-1]
        (
            stride,
            padding,
            dilation,
            groups,
            x_scale,
            x_zp,
            accum_scale,
            accum_zp,
            o_scale,
            o_zp,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ) = const_args[-16:]
        conv_args = (
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
            o_scale,
            o_zp,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            conv_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self):
        return [self.inputs[self.idx_for_inplace_sum].get_name()]

    def get_unbacked_symbol_defs(self) -> Set[sympy.Symbol]:
        return set()

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale,
        x_zero_point,
        qaccum: "TensorBox",
        accum_scale,
        accum_zero_point,
        qw: "TensorBox",  # packed_weight
        w_scale,
        w_zero_point,
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_scale: "TensorBox",
        output_zero_point: "TensorBox",
        output_dtype,
        binary_attr,
        alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
    ):
        transposed = False
        output_padding = None
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls,
            qx,
            qw,
            bias,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            output_padding,
        )

        qaccum = cls.require_stride_order(qaccum, req_stride_order)
        inputs.append(qaccum)

        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [w_scale, w_zero_point]
        constant_args = constant_args + [
            x_scale,
            x_zero_point,
            accum_scale,
            accum_zero_point,
            output_scale,
            output_zero_point,
            output_dtype,
            binary_attr,
            alpha,
            unary_attr,
            may_convert_to_optional(unary_scalars),
            unary_algorithm,
        ]

        assert (
            binary_attr == "sum"
        ), "For now, only post op sum is supported in QConvPointWiseBinaryPT2E."

        V.graph.mark_buffer_mutated(qaccum.get_name())
        packed = QConvPointWiseBinaryPT2E(
            layout=NoneLayout(qaccum.get_device()),
            inputs=inputs,
            constant_args=constant_args,
        )

        # Return accum since it has been inplace changed.
        return packed.inputs[packed.idx_for_inplace_sum]


class MKLPackedLinear(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkl._mkl_linear",
            cpp_kernel_name="mkl::_mkl_linear",
        )
        self.cpp_kernel_key = "mkl_linear"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& self,
                const at::Tensor& mkl_weight_t,
                const at::Tensor& origin_weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                const int64_t prepack_batch_size)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    def create(cls, x, packed_w, orig_w, B, batch_size):
        x = cls.require_stride1(cls.realize_input(x))
        orig_w = cls.require_stride1(cls.realize_input(orig_w))
        *m, _ = x.get_size()
        oc, _ = orig_w.get_size()
        output_size = list(m) + [oc]
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        inputs = [x, packed_w, orig_w]
        constant_args = [batch_size]
        if B is not None:
            inputs += [B]
        else:
            constant_args.insert(0, None)

        return MKLPackedLinear(
            layout=FixedLayout(
                x.get_device(), x.get_dtype(), output_size, output_stride
            ),
            inputs=inputs,
            constant_args=constant_args,
        )


class LinearUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._linear_pointwise",
            cpp_kernel_name="mkldnn::_linear_pointwise",
        )
        self.cpp_kernel_key = "linear_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    def create(cls, x, w, B, attr, scalars, algorithm):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, None)

        return LinearUnary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="torch.ops.mkldnn._linear_pointwise.binary",
            cpp_kernel_name="mkldnn::_linear_pointwise",
        )
        self.cpp_kernel_overload_name = "binary"
        self.cpp_kernel_key = "linear_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr)
        """

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )

    @classmethod
    def create(cls, x, y, w, B, attr):
        x = cls.require_contiguous(cls.realize_input(x))
        y = cls.require_contiguous(cls.realize_input(y))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()

        inputs = [x, y, w]
        constant_args = [attr]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, B)

        return LinearBinary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class QLinearPointwisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
        x_scale_zp_are_tensors=False,
    ):
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp]
            - const_args is: [x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp]
            - const_args is: [bias, x_scale, x_zp, o_scale, o_zp,
              fp32_output, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = has_bias
        self.x_scale_zp_are_tensors = x_scale_zp_are_tensors
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name=(
                "torch.ops.onednn.qlinear_pointwise.tensor"
                if x_scale_zp_are_tensors
                else "torch.ops.onednn.qlinear_pointwise.default"
            ),
            cpp_kernel_name="onednn::qlinear_pointwise",
        )
        self.cpp_kernel_overload_name = "tensor" if x_scale_zp_are_tensors else ""
        self.cpp_kernel_key = "qlinear_pointwise"
        x_scale_type_str, x_zp_type_str = (
            ("at::Tensor", "at::Tensor")
            if x_scale_zp_are_tensors
            else ("double", "int64_t")
        )
        self.cpp_op_schema = f"""
            at::Tensor(
                at::Tensor act,
                {x_scale_type_str} act_scale,
                {x_zp_type_str} act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> bias,
                double output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                c10::string_view post_op_name,
                torch::List<c10::optional<at::Scalar>> post_op_args,
                c10::string_view post_op_algorithm)"""

    def codegen(self, wrapper):
        # Parser the inputs and constant
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())

        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        w_scale, w_zp = args[-2], args[-1]
        if self.x_scale_zp_are_tensors:
            assert len(args) >= 4
            x_scale, x_zp = args[-4], args[-3]
            (
                o_scale,
                o_zp,
                output_dtype,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-6:]
        else:
            assert len(const_args) >= 8
            (
                x_scale,
                x_zp,
                o_scale,
                o_zp,
                output_dtype,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-8:]

        codegen_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            bias,
            o_scale,
            o_zp,
            output_dtype,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: float,
        x_zero_point: int,
        qw: "TensorBox",  # packed_weight
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        bias: "TensorBox",
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ):
        (inputs, constant_args, kernel_layout, _) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
        )

        if isinstance(x_scale, TensorBox) and isinstance(x_zero_point, TensorBox):
            x_scale.realize()
            x_zero_point.realize()
            inputs = inputs + [x_scale, x_zero_point]
            x_scale_zp_are_tensors = True
        else:
            assert isinstance(x_scale, float) and isinstance(x_zero_point, int)
            constant_args = constant_args + [x_scale, x_zero_point]
            x_scale_zp_are_tensors = False
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [w_scale, w_zero_point]
        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            post_op_name,
            may_convert_to_optional(post_op_args),
            post_op_algorithm,
        ]

        assert output_dtype is not None
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_linear_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set fp32_output, the output buf should be dtype float32 instead of uint8.
            kernel_layout.dtype = output_dtype

        return QLinearPointwisePT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            has_bias=(bias is not None),
            x_scale_zp_are_tensors=x_scale_zp_are_tensors,
        )


class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
        x_scale_zp_are_tensors=False,
    ):
        """
        if bias is not None
            - inputs = [x, w, b, weight_scale, weight_zp, x2]
            - const_args is: [x_scale, x_zp, o_scale, o_zp,
              fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, weight_scale, weight_zp, x2]
            - const_args is: [bias, x_scale, x_zp, o_scale, o_zp,
              fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = has_bias
        self.x_scale_zp_are_tensors = x_scale_zp_are_tensors
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name=(
                "torch.ops.onednn.qlinear_pointwise.binary_tensor"
                if x_scale_zp_are_tensors
                else "torch.ops.onednn.qlinear_pointwise.binary"
            ),
            cpp_kernel_name="onednn::qlinear_pointwise",
        )
        self.cpp_kernel_overload_name = (
            "binary_tensor" if x_scale_zp_are_tensors else "binary"
        )
        self.cpp_kernel_key = "qlinear_pointwise_binary"
        x_scale_type_str, x_zp_type_str = (
            ("at::Tensor", "at::Tensor")
            if x_scale_zp_are_tensors
            else ("double", "int64_t")
        )
        self.cpp_op_schema = f"""
            at::Tensor(
                at::Tensor act,
                {x_scale_type_str} act_scale,
                {x_zp_type_str} act_zero_point,
                at::Tensor weight,
                at::Tensor weight_scales,
                at::Tensor weight_zero_points,
                c10::optional<at::Tensor> other,
                c10::optional<at::Tensor> bias,
                double inv_output_scale,
                int64_t output_zero_point,
                c10::optional<c10::ScalarType> output_dtype,
                double other_scale,
                int64_t other_zero_point,
                c10::string_view binary_post_op,
                double binary_alpha,
                c10::string_view unary_post_op,
                torch::List<c10::optional<at::Scalar>> unary_post_op_args,
                c10::string_view unary_post_op_algorithm)"""

    def codegen(self, wrapper):
        # Parser the inputs and constant
        args = [x.codegen_reference() for x in self.inputs]
        const_args = []
        const_args.extend(self.codegen_const_args())

        x = args[0]
        packed_weight = args[1]
        bias = args[2] if self.has_bias else const_args[0]
        w_scale, w_zp, other = args[-3], args[-2], args[-1]
        if self.x_scale_zp_are_tensors:
            assert len(args) >= 5
            x_scale, x_zp = args[-5], args[-4]
            (
                o_scale,
                o_zp,
                output_dtype,
                other_scale,
                other_zp,
                binary_attr,
                alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-10:]
        else:
            assert len(const_args) >= 8
            (
                x_scale,
                x_zp,
                o_scale,
                o_zp,
                output_dtype,
                other_scale,
                other_zp,
                binary_attr,
                alpha,
                unary_attr,
                unary_scalars,
                unary_algorithm,
            ) = const_args[-12:]

        codegen_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            other,
            bias,
            o_scale,
            o_zp,
            output_dtype,
            other_scale,
            other_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        )
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.python_kernel_name,
            self.cpp_kernel_name,
            codegen_args,
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overload_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self):
        binary_post_op = self.constant_args[-5]
        if binary_post_op == "sum":
            return [self.inputs[-1].get_name()]
        else:
            return []

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: float,
        x_zero_point: int,
        qw: "TensorBox",  # packed_weight
        w_scale: "TensorBox",
        w_zero_point: "TensorBox",
        other: "TensorBox",
        bias: "TensorBox",
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        other_scale,
        other_zp,
        binary_post_op,
        binary_alpha,
        unary_post_op,
        unary_post_op_args,
        unary_post_op_algorithm,
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
        )

        if isinstance(x_scale, TensorBox) and isinstance(x_zero_point, TensorBox):
            x_scale.realize()
            x_zero_point.realize()
            inputs = inputs + [x_scale, x_zero_point]
            x_scale_zp_are_tensors = True
        else:
            assert isinstance(x_scale, float) and isinstance(x_zero_point, int)
            constant_args = constant_args + [x_scale, x_zero_point]
            x_scale_zp_are_tensors = False
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [w_scale, w_zero_point]
        if binary_post_op == "sum":
            other = cls.require_stride_order(other, req_stride_order)
        inputs.append(other)
        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            other_scale,
            other_zp,
            binary_post_op,
            binary_alpha,
            unary_post_op,
            may_convert_to_optional(unary_post_op_args),
            unary_post_op_algorithm,
        ]

        if binary_post_op == "sum":
            V.graph.mark_buffer_mutated(other.get_name())
            packed = QLinearPointwiseBinaryPT2E(
                layout=NoneLayout(other.get_device()),
                inputs=inputs,
                constant_args=constant_args,
                has_bias=(bias is not None),
                x_scale_zp_are_tensors=x_scale_zp_are_tensors,
            )
            # Return other since it has been inplace changed.
            return packed.inputs[-1]

        assert output_dtype is not None
        if output_dtype in [torch.float32, torch.bfloat16]:
            # in _prepare_linear_fusion_create, we use x.dtype (uint8) to create kernel_layout
            # if we set fp32_output, the output buf should be dtype float32 instead of uint8.
            kernel_layout.dtype = output_dtype

        return QLinearPointwiseBinaryPT2E(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            has_bias=(bias is not None),
            x_scale_zp_are_tensors=x_scale_zp_are_tensors,
        )


class MkldnnRnnLayer(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            python_kernel_name="aten.mkldnn_rnn_layer",
            cpp_kernel_name="at::mkldnn_rnn_layer",
        )

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        w0: "TensorBox",
        w1: "TensorBox",
        w2: "TensorBox",
        w3: "TensorBox",
        hx: "TensorBox",
        cx: "TensorBox",
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
        x = cls.require_stride1(cls.realize_input(x))
        # If batch_first, x has been permuted in lstm before entering the mkldnn_rnn_layer.
        # Make sure x is contiguous in batch_first case.
        x.freeze_layout()
        w0 = cls.require_stride1(cls.realize_input(w0))
        w1 = cls.require_stride1(cls.realize_input(w1))
        w2 = cls.require_stride1(cls.realize_input(w2))
        w3 = cls.require_stride1(cls.realize_input(w3))
        hx = cls.require_stride1(cls.realize_input(hx))
        hx.freeze_layout()
        cx = cls.require_stride1(cls.realize_input(cx))
        cx.freeze_layout()

        input_size = x.get_size()
        assert len(input_size) == 3, "Expect lstm input to be 3D"
        # batch_first is handled in the lstm OP. When entering
        # rnn_layer here, we'll always have batch_first = False
        seq_length, mini_batch, input_size = input_size
        output_shape = [seq_length, mini_batch, hidden_size]

        hy_shape = hx.get_size()
        cy_shape = cx.get_size()

        res: List[IRNode] = []

        inputs = [x, w0, w1, w2, w3, hx, cx]
        constant_args = [
            reverse,
            batch_sizes,
            mode,
            hidden_size,
            num_layers,
            has_biases,
            bidirectional,
            batch_first,
            train,
        ]

        packed = MkldnnRnnLayer(
            MultiOutputLayout(x.get_device()),
            inputs=inputs,
            constant_args=constant_args,
        )

        def get_strides_of_lstm_output(output_shape, batch_first):
            assert len(output_shape) == 3, "Expect output_shape to be 3D"
            return FlexibleLayout.contiguous_strides(output_shape)

        output_sizes = [output_shape, hy_shape, cy_shape]
        output_strides = [
            get_strides_of_lstm_output(output_shape, batch_first),
            FlexibleLayout.contiguous_strides(hy_shape),
            FlexibleLayout.contiguous_strides(cy_shape),
        ]
        output_ir = [
            MultiOutput(
                FixedLayout(
                    x.get_device(),
                    x.get_dtype(),
                    output_size,
                    output_stride,
                ),
                packed,
                [(tuple, i)],
            )
            for i, (output_size, output_stride) in enumerate(
                zip(output_sizes, output_strides)
            )
        ]

        return output_ir
