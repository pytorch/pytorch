# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestReshapeConverter(AccTestCase):
    @parameterized.expand(
        [
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    def test_reshape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(TestModule(target_shape), inputs, expected_ops={acc_ops.reshape})

    @parameterized.expand(
        [
            ((-1, 2),),
            ((1, 2, -1),),
        ]
    )
    def test_reshape_with_dynamic_shape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(target_shape), input_specs, expected_ops={acc_ops.reshape}
        )

if __name__ == '__main__':
    run_tests()
