# Owner(s): ["oncall: fx"]

import torch
import torch.fx
from torch.testing._internal.common_fx2trt import VanillaTestCase
from parameterized import parameterized, param
from torch.testing._internal.common_utils import run_tests


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

if __name__ == '__main__':
    run_tests()
