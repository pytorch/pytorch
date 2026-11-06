# Owner(s): ["module: intel"]

import itertools
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize as parametrize_test,
    run_tests,
    TestCase,
)


CONV_MODULES = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


def _binary_add_args(conv):
    # trailing args of mkldnn._convolution_pointwise(_).binary after input/other
    return (
        conv.weight,
        conv.bias,
        conv.padding,
        conv.stride,
        conv.dilation,
        conv.groups,
        "add",
        None,
        None,
        [],
        None,
    )


class PointwisePostOp(NamedTuple):
    attr: str
    pointwise_module: nn.Module
    scalars: list = []
    algorithm: str = ""


class TestoneDNNFusion(TestCase):
    def _unary_list(self):
        unary_list = {
            "relu": PointwisePostOp("relu", nn.ReLU()),
            "sigmoid": PointwisePostOp("sigmoid", nn.Sigmoid()),
            "tanh": PointwisePostOp("tanh", nn.Tanh()),
            "hardswish": PointwisePostOp("hardswish", nn.Hardswish()),
            "swish": PointwisePostOp("swish", nn.SiLU()),
            "leaky_relu": PointwisePostOp(
                "leaky_relu", nn.LeakyReLU(0.1, inplace=False), scalars=[0.1]
            ),
            "hardtanh": PointwisePostOp(
                "hardtanh",
                nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False),
                scalars=[-0.5, 4],
            ),
            "gelu_none": PointwisePostOp(
                "gelu", nn.GELU(approximate="none"), algorithm="none"
            ),
            "gelu_tanh": PointwisePostOp(
                "gelu", nn.GELU(approximate="tanh"), algorithm="tanh"
            ),
        }
        return unary_list

    def _binary_list(self):
        binary_list = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
        }
        return binary_list

    def test_linear_unary_fusion_ops(self, device):
        class M(nn.Module):
            def __init__(self, unary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.unary = unary_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.unary(x)
                return x

        for pointwise_info in self._unary_list().values():
            options = itertools.product([[2, 3, 10], [2, 10]], [True, False])
            for input_shape, bias in options:
                with torch.no_grad():
                    mod = M(
                        pointwise_info.pointwise_module, input_shape[-1], 10, bias
                    ).eval()
                    mod = mod.to(device)
                    v = torch.randn(input_shape)
                    v = v.to(device)
                    ref = mod(v)
                    attr = pointwise_info.attr
                    scalars = pointwise_info.scalars
                    algorithm = pointwise_info.algorithm
                    fused = torch.ops.mkldnn._linear_pointwise(
                        v,
                        mod.linear.weight,
                        mod.linear.bias,
                        attr,
                        scalars,
                        algorithm,
                    )
                    self.assertEqual(ref, fused)

    def test_linear_binary_fusion_ops(self, device):
        class M(nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary = binary_fn

            def forward(self, x, other):
                x = self.linear(x)
                x = self.binary(x, other)
                return x

        out_feature = 20
        in_feature = 10
        for pointwise_name, pointwise_fn in self._binary_list().items():
            with torch.no_grad():
                input = torch.randn(4, in_feature).to(device)
                model = M(pointwise_fn, in_feature, out_feature, True).eval().to(device)
                other = torch.randn(4, out_feature).to(device)
                ref = model(input, other)
                attr = pointwise_name
                fused = torch.ops.mkldnn._linear_pointwise(
                    input,
                    other,
                    model.linear.weight,
                    model.linear.bias,
                    attr,
                )
                self.assertEqual(ref, fused)

    def test_conv_unary_fusion_ops(self):
        class M(nn.Module):
            def __init__(
                self,
                unary_fn,
                dim,
                in_channels,
                out_channels,
                dilation,
                groups,
                bias,
                **kwargs,
            ):
                super().__init__()
                self.conv = CONV_MODULES[dim](
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    **kwargs,
                )
                self.unary = unary_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.unary(x)
                return x

        input_shapes = {2: (112, 112), 3: (55, 55, 55)}
        for pointwise_info in self._unary_list().values():
            for dim in [2, 3]:
                channels_last = (
                    torch.channels_last if dim == 2 else torch.channels_last_3d
                )
                options = itertools.product(
                    [True, False],
                    [1, 2],
                    [1, 4],
                    [torch.contiguous_format, channels_last],
                )
                for bias, dilation, groups, memory_format in options:
                    oC = 32 * groups
                    iC = 3 * groups
                    x_shape = (1, iC) + input_shapes[dim]
                    x = torch.randn(x_shape, dtype=torch.float32).to(
                        memory_format=memory_format
                    )
                    mod = M(
                        pointwise_info.pointwise_module,
                        dim,
                        iC,
                        oC,
                        dilation,
                        groups,
                        bias,
                        kernel_size=3,
                    )
                    mod = mod.to(memory_format=memory_format).eval()
                    with torch.no_grad():
                        x = x.to("xpu")
                        mod = mod.to("xpu")
                        ref = mod(x)
                        attr = pointwise_info.attr
                        scalars = pointwise_info.scalars
                        algorithm = pointwise_info.algorithm
                        fused = torch.ops.mkldnn._convolution_pointwise(
                            x,
                            mod.conv.weight,
                            mod.conv.bias,
                            mod.conv.padding,
                            mod.conv.stride,
                            mod.conv.dilation,
                            mod.conv.groups,
                            attr,
                            scalars,
                            algorithm,
                        )
                    self.assertEqual(ref, fused)

    def test_conv_binary_fusion_ops(self):
        class M(nn.Module):
            def __init__(
                self,
                binary_fn,
                dim,
                in_channels,
                out_channels,
                dilation,
                groups,
                bias,
                **kwargs,
            ):
                super().__init__()
                self.conv = CONV_MODULES[dim](
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    **kwargs,
                )
                self.binary = binary_fn

            def forward(self, x, other):
                x = self.conv(x)
                x = self.binary(x, other)
                return x

        for pointwise_name, pointwise_fn in self._binary_list().items():
            x = torch.randn(
                (
                    1,
                    3,
                    112,
                    112,
                )
            ).to("xpu")
            mod = M(pointwise_fn, 2, 3, 3, 1, 1, True, kernel_size=3).to("xpu")
            other = torch.randn_like(mod.conv(x))
            with torch.no_grad():
                ref = mod(x, other)
                unary_attr = None
                attr = pointwise_name
                fused = torch.ops.mkldnn._convolution_pointwise(
                    x,
                    other,
                    mod.conv.weight,
                    mod.conv.bias,
                    mod.conv.padding,
                    mod.conv.stride,
                    mod.conv.dilation,
                    mod.conv.groups,
                    attr,
                    None,
                    unary_attr,
                    [],
                    None,
                )
                if attr == "add":
                    fused_inplace = torch.ops.mkldnn._convolution_pointwise_(
                        other,
                        x,
                        mod.conv.weight,
                        mod.conv.bias,
                        mod.conv.padding,
                        mod.conv.stride,
                        mod.conv.dilation,
                        mod.conv.groups,
                        attr,
                        None,
                        unary_attr,
                        [],
                        None,
                    )
                    self.assertEqual(ref, other)
                    self.assertEqual(ref, fused_inplace)
                self.assertEqual(ref, fused, atol=5e-4, rtol=5e-4)

    @parametrize_test("inplace", [False, True])
    @parametrize_test("x_channels_last", [False, True])
    @parametrize_test(
        "other_layout",
        ["contiguous", "channels_last", "sliced", "channels_last_sliced", "offset"],
    )
    def test_conv_binary_other_layout(
        self, device, inplace, x_channels_last, other_layout
    ):
        # `other` is read (and for the in-place op written) in its own layout,
        # which may differ from the one the conv picked from src/weight and may
        # not even be dense
        conv = nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        if x_channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            ref = conv(x)
            sliced = other_layout.endswith("sliced")
            other = torch.randn_like(ref).repeat(1, 1, 1, 2 if sliced else 1)
            if other_layout.startswith("channels_last"):
                other = other.to(memory_format=torch.channels_last)
            if sliced:
                other = other[..., ::2]
            if other_layout == "offset":
                # not 64-byte aligned
                buf = torch.empty(other.numel() + 1, device=device)
                other = buf[1:].view_as(other).copy_(other)
            expected = ref + other
            args = _binary_add_args(conv)
            if inplace:
                fused = torch.ops.mkldnn._convolution_pointwise_.binary(other, x, *args)
                self.assertEqual(expected, other)
            else:
                fused = torch.ops.mkldnn._convolution_pointwise.binary(x, other, *args)
        self.assertEqual(expected, fused)

    @parametrize_test("x_channels_last", [False, True])
    @parametrize_test("expanded", [False, True])
    def test_conv_binary_broadcast_other(self, device, x_channels_last, expanded):
        conv = nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        if x_channels_last:
            x = x.to(memory_format=torch.channels_last)
        args = _binary_add_args(conv)
        with torch.no_grad():
            ref = conv(x)
            other = torch.randn(1, 6, 1, 1, device=device)
            if expanded:
                other = other.expand_as(ref)
            fused = torch.ops.mkldnn._convolution_pointwise.binary(x, other, *args)
        self.assertEqual(ref + other, fused)

    def test_conv_binary_inplace_overlapping_other(self, device):
        conv = nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        args = _binary_add_args(conv)
        other = torch.zeros(1, 6, 1, 1, device=device).expand(2, 6, 6, 6)
        with (
            torch.no_grad(),
            self.assertRaisesRegex(RuntimeError, "more than one element"),
        ):
            torch.ops.mkldnn._convolution_pointwise_.binary(other, x, *args)

    @parametrize_test("aliased", ["input", "weight", "bias"])
    def test_conv_binary_inplace_other_aliases_operand(self, device, aliased):
        conv = nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        operand = {"input": x, "weight": conv.weight, "bias": conv.bias}[aliased]
        buf = torch.zeros(2 * 6 * 6 * 6 + operand.numel(), device=device)
        with torch.no_grad():
            operand.set_(buf[: operand.numel()].view_as(operand).copy_(operand))
        other = buf[: 2 * 6 * 6 * 6].view(2, 6, 6, 6)
        args = _binary_add_args(conv)
        with (
            torch.no_grad(),
            self.assertRaisesRegex(RuntimeError, "refer to a single memory location"),
        ):
            torch.ops.mkldnn._convolution_pointwise_.binary(other, x, *args)

    def test_conv_binary_self_overlapping_other(self, device):
        # overlapping nonzero strides can't be described to oneDNN, so other is
        # read from a contiguous copy
        conv = nn.Conv2d(3, 6, 3).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        other = torch.randn(2 * 6 * 6 * 6, device=device).as_strided(
            (2, 6, 6, 6), (216, 6, 6, 1)
        )
        args = _binary_add_args(conv)
        with torch.no_grad():
            fused = torch.ops.mkldnn._convolution_pointwise.binary(x, other, *args)
            self.assertEqual(conv(x) + other, fused)


instantiate_device_type_tests(
    TestoneDNNFusion, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
