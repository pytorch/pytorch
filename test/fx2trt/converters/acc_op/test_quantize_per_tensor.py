# Owner(s): ["oncall: fx"]

import unittest

import tensorrt as trt
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


@unittest.skip(
    """
    Tests related to quantize have issue creating engine, disable now.
    """
)
@unittest.skipIf(
    trt.__version__ < "8.0",
    "Explicit quantization only supported in TensorRT 8.0 and later",
)
class TestQuantizePerTensorConverter(AccTestCase):
    def test_quantize_per_tensor(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.quantize_per_tensor(x, 1, 0, torch.quint8)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.quantize_per_tensor})

    def test_quantize_per_tensor_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.quantize_per_tensor(x, 1, 0, torch.quint8)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.quantize_per_tensor}
        )

if __name__ == '__main__':
    run_tests()
