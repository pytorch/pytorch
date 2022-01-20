# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from parameterized import parameterized, param
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestAvgPoolConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            ("kernal_size", 3),
            ("stride", 1, 2),
            ("tuple_parameters", 2, (1, 1), (1, 1)),
            param("padding", 2, padding=1),
            param("ceil_mode", 1, ceil_mode=True),
            param("include_pad", 2, padding=1, count_include_pad=False),
        ]
    )
    def test_avg_pool2d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

            def forward(self, x):
                return self.avg_pool(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.avg_pool2d})

if __name__ == '__main__':
    run_tests()
