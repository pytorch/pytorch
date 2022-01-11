# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized, param
from torch.testing._internal.common_utils import run_tests


class TestClampConverter(AccTestCase):
    @parameterized.expand(
        [
            param("default", min=-1, max=0),
            param("min", min=0.5),
            param("max", max=0.5),
            param("minBiggerThanMax", min=1, max=0),
        ]
    )
    def test_clamp(
        self,
        test_name,
        min=None,
        max=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min, max)

        inputs = [torch.randn(3, 4)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.clamp})

if __name__ == '__main__':
    run_tests()
