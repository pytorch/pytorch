# mypy: allow-untyped-defs
from typing import Any, List, Optional

import sympy

import torch
from torch._prims_common import make_channels_last_strides_for
from torch.utils._ordered_set import OrderedSet

from .ir import (
    ExternKernelAlloc,
    FixedLayout,
    FlexibleLayout,
    get_device_type,
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
    quantize_args: Optional[List["TensorBox"]] = None,
    other: Optional["TensorBox"] = None,
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
            weight_size.extend(prepacked_weight_size[d] for d in range(2, dim))
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

    assert get_device_type(x) == get_device_type(weight)
    assert get_device_type(x) in ["cpu", "xpu"]
    inputs = [x]

    if quantize_args is not None:
        x_scale, x_zero_point, w_scale, w_zero_point = quantize_args
        x_scale.realize()
        x_zero_point.realize()
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [x_scale, x_zero_point] + [weight] + [w_scale, w_zero_point]
    else:
        inputs += [weight]

    if other is not None:
        other = cls.require_stride_order(other, req_stride_order)
        assert isinstance(other, TensorBox)
        inputs += [other]

    kernel_layout = FixedLayout(
        x.get_device_or_error(),
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
    return inputs, constant_args, kernel_layout, req_stride_order, other


def _prepare_linear_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    quantize_args: Optional[List["TensorBox"]] = None,
    other: Optional["TensorBox"] = None,
    binary_sum: bool = False,
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
    assert x.get_device().type in ["cpu", "xpu"] and weight.get_device().type in [
        "cpu",
        "xpu",
    ]
    assert x.get_device().type == weight.get_device().type
    inputs = [x]

    if quantize_args is not None:
        x_scale, x_zero_point, w_scale, w_zero_point = quantize_args
        x_scale.realize()
        x_zero_point.realize()
        w_scale.realize()
        w_zero_point.realize()
        inputs = inputs + [x_scale, x_zero_point] + [weight] + [w_scale, w_zero_point]
    else:
        inputs += [weight]

    if other is not None:
        if binary_sum:
            other = cls.require_stride_order(other, req_stride_order)
        inputs = inputs + [other]

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
    return inputs, constant_args, kernel_layout, req_stride_order, other


def _create_output_node(packed):
    output_ir = MultiOutput(
        packed.get_layout(),
        packed,
        [],
    )
    packed.layout = MultiOutputLayout(device=packed.get_device())
    packed.outputs = [output_ir]
    return output_ir


class ConvolutionUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise.default,
            cpp_kernel_name="aoti_torch_cpu_mkldnn__convolution_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

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
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
            _,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(scalars),
            algorithm,
        ]
        packed = ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise.binary,
            cpp_kernel_name="aoti_torch_cpu_mkldnn__convolution_pointwise_binary",
        )
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

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
            _,
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
        packed = ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ) -> None:
        # Due to constrain of op.call, other (Tensor&) should be at input[0]
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        super().__init__(
            kernel_layout,
            reordered_inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_pointwise_.binary,
            cpp_kernel_name="aoti_torch_cpu_mkldnn__convolution_pointwise_binary_",
        )

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=inputs[0].get_device()), inputs[0], self),
            MutationOutput(NoneLayout(device=inputs[1].get_device()), inputs[1], self),
        ]

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

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
            _,
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
            kernel_layout=NoneLayout(device=inputs[1].get_device()),  # type: ignore[arg-type]
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
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._convolution_transpose_pointwise.default,
            cpp_kernel_name="aoti_torch_cpu_mkldnn__convolution_transpose_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

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
        packed = ConvolutionTransposeUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)


class QConvPointWisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
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
            op_overload=torch.ops.onednn.qconv2d_pointwise.default,
            cpp_kernel_name="aoti_torch_cpu__qconv2d_pointwise_tensor",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
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
        (
            inputs,
            constant_args,
            kernel_layout,
            _,
            _,
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
            [x_scale, x_zero_point, w_scale, w_zero_point],
        )
        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        constant_args = constant_args + [
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
    ) -> None:
        """
        Needs input/weight/output qparams
        if bias is not None
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum, b]
            - const_args = [stride, padding, dilation, groups, o_scale, o_zp,
            output_dtype, accum_scale, accum_zp, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, x_scale, x_zp, w,  w_scale, w_zp, accum]
            - const_args [b, stride, padding, dilation, groups, o_scale, o_zp,
             output_dtype, accum_scale, accum_zp, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = len(inputs) == 8
        self.idx_for_inplace_sum = 6
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.onednn.qconv2d_pointwise.binary,
            cpp_kernel_name=("aoti_torch_cpu__qconv2d_pointwise_binary_tensor"),
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self):
        return [self.inputs[self.idx_for_inplace_sum].get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
        qw: "TensorBox",  # packed_weight
        w_scale,
        w_zero_point,
        qaccum: "TensorBox",
        bias: "TensorBox",
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_scale: "TensorBox",
        output_zero_point: "TensorBox",
        output_dtype,
        accum_scale,
        accum_zero_point,
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
            qaccum,
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
            [x_scale, x_zero_point, w_scale, w_zero_point],
            qaccum,
        )

        # swap padding and stride to align with functional conv arg order
        if bias is None:
            constant_args[1], constant_args[2] = constant_args[2], constant_args[1]
        else:
            constant_args[0], constant_args[1] = constant_args[1], constant_args[0]

        constant_args = constant_args + [
            output_scale,
            output_zero_point,
            output_dtype,
            accum_scale,
            accum_zero_point,
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
            layout=NoneLayout(device=qaccum.get_device()),
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
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkl._mkl_linear.default,
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

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
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._linear_pointwise.default,
            cpp_kernel_name="aoti_torch_cpu__linear_pointwise",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

    @classmethod
    def create(cls, x, w, B, attr, scalars, algorithm):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()
        output_size = list(m) + [oc]
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, None)

        packed = LinearUnary(
            layout=FixedLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.mkldnn._linear_pointwise.binary,
            cpp_kernel_name="aoti_torch_cpu__linear_pointwise_binary",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

    @classmethod
    def create(cls, x, y, w, B, attr):
        x = cls.require_contiguous(cls.realize_input(x))
        y = cls.require_contiguous(cls.realize_input(y))
        w = cls.require_contiguous(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()
        output_size = list(m) + [oc]
        output_stride = FlexibleLayout.contiguous_strides(output_size)
        inputs = [x, y, w]
        constant_args = [attr]
        if B is not None:
            B = cls.require_contiguous(cls.realize_input(B))
            inputs.append(B)
        else:
            constant_args.insert(0, B)

        packed = LinearBinary(
            layout=FixedLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=output_size,
            ),
            inputs=inputs,
            constant_args=constant_args,
        )
        return _create_output_node(packed)

    def apply_constraint(self):
        pass


class QLinearPointwisePT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
    ) -> None:
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
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=(torch.ops.onednn.qlinear_pointwise.tensor),
            cpp_kernel_name=("aoti_torch_cpu__qlinear_pointwise_tensor"),
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
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
        (inputs, constant_args, kernel_layout, _, _) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
            [x_scale, x_zero_point, w_scale, w_zero_point],
        )

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
        )


class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        has_bias=True,
    ) -> None:
        """
        if bias is not None
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2, bias]
            - const_args is: [o_scale, o_zp,
              fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        else
            - inputs = [x, w, x_scale, x_zp, weight_scale, weight_zp, x2]
            - const_args is: [bias, o_scale, o_zp,
              fp32_output, binary_attr, aplha, unary_attr, unary_scalars, unary_algorithm]
        """
        self.has_bias = has_bias
        self.idx_for_inplace_sum = 6
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=(torch.ops.onednn.qlinear_pointwise.binary_tensor),
            cpp_kernel_name="aoti_torch_cpu__qlinear_pointwise_binary_tensor",
        )

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        super().codegen(wrapper)
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_mutation_names(self):
        binary_post_op = self.constant_args[-5]
        if binary_post_op == "sum":
            return [self.inputs[self.idx_for_inplace_sum].get_name()]
        else:
            return []

    @classmethod
    def create(
        cls,
        qx: "TensorBox",
        x_scale: "TensorBox",
        x_zero_point: "TensorBox",
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
            other,
        ) = _prepare_linear_fusion_create(
            cls,
            qx,
            qw,
            bias,
            [x_scale, x_zero_point, w_scale, w_zero_point],
            other,
            binary_post_op == "sum",
        )

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
                layout=NoneLayout(device=other.get_device()),
                inputs=inputs,
                constant_args=constant_args,
                has_bias=(bias is not None),
            )
            # Return other since it has been inplace changed.
            return packed.inputs[packed.idx_for_inplace_sum]

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
        )


class MkldnnRnnLayer(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ) -> None:
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            op_overload=torch.ops.aten.mkldnn_rnn_layer.default,
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
            MultiOutputLayout(device=x.get_device()),
            inputs=inputs,
            constant_args=constant_args,
        )

        def get_strides_of_lstm_output(output_shape, batch_first):
            assert len(output_shape) == 3, "Expect output_shape to be 3D"
            return FlexibleLayout.contiguous_strides(output_shape)

        # C shim call requires all the outputs to be passed in, and thus the last
        # dummy return value is added.
        output_sizes = [output_shape, hy_shape, cy_shape, [1]]
        output_strides = [
            get_strides_of_lstm_output(output_shape, batch_first),
            FlexibleLayout.contiguous_strides(hy_shape),
            FlexibleLayout.contiguous_strides(cy_shape),
            [1],
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
        packed.outputs = output_ir

        return output_ir

    def codegen(self, wrapper):
        wrapper.include_extra_header("torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h")
        return super().codegen(wrapper)
