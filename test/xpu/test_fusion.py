# Owner(s): ["module: intel"]

import itertools
from typing import List, NamedTuple

import torch
import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


CONV_MODULES = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class PointwisePostOp(NamedTuple):
    attr: str
    pointwise_module: nn.Module
    scalars: List = []
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

    def test_linear_unary_fusion_ops(self):
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
                    mod = mod.to("xpu")
                    v = torch.randn(input_shape)
                    v = v.to("xpu")
                    ref = mod(v)
                    attr = pointwise_info.attr
                    scalars = pointwise_info.scalars
                    algorithm = pointwise_info.algorithm
                    fused = torch.ops.mkldnn._linear_pointwise(
                        v,
                        mod.linear.weight.transpose(1, 0),
                        mod.linear.bias,
                        attr,
                        scalars,
                        algorithm,
                    )
                    self.assertEqual(ref, fused)

    def test_linear_binary_fusion_ops(self):
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
                input = torch.randn(4, in_feature).xpu()
                model = M(pointwise_fn, in_feature, out_feature, True).eval().xpu()
                other = torch.randn(4, out_feature).xpu()
                ref = model(input, other)
                attr = pointwise_name
                fused = torch.ops.mkldnn._linear_pointwise(
                    input,
                    other,
                    model.linear.weight.transpose(1, 0),
                    model.linear.bias,
                    attr,
                )
                self.assertEqual(ref, fused)


instantiate_device_type_tests(
    TestoneDNNFusion, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
