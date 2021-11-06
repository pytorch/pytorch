# Owner(s): ["oncall: fx"]

import torch
import torch.fx
from caffe2.torch.fb.fx2trt.tests.test_utils import VanillaTestCase
from parameterized import parameterized, param


class TestConvolutionConverter(VanillaTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1, 1), (0, 0)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
        ]
    )
    def test_conv2d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 6, kernel_size, stride, padding, dilation, groups, bias
                )

            def forward(self, x):
                return self.conv(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={torch.nn.modules.conv.Conv2d})
