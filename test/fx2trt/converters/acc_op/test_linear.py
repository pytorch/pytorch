# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase, InputTensorSpec
from parameterized import parameterized


class TestLinearConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default"),
            ("no_bias", False),
        ]
    )
    def test_linear(
        self,
        test_name,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256, bias)

            def forward(self, x):
                return self.linear(x)

        inputs = [torch.randn(1, 512)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.linear})

    def test_linear_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(512, 256)

            def forward(self, x):
                return self.linear(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, 512),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 512), (3, 3, 512), (4, 3, 512))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={acc_ops.linear},
        )
