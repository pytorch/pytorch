# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.linear import LinearAPoT
from torch.nn.modules.linear import Linear
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestNonUniformObserver(TestCase):
    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (4,4) and k=1
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def test_linear_APoT_k1(self, device):
        # weight: fp tensor
        weight = 1000 * torch.rand(4, 4, device=device)

        # activation: fp32 tensor with ~ integer values
        activation = torch.randint(
            low=0, high=255, size=(4, 4), dtype=torch.float, device=device
        )

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 1)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False, device=device)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.to(dtype=torch.float)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertEqual(apot_linear_result, fp_linear_result, rtol=0, atol=0)

    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (5,3), (3, 5) and k=2
    """
    def test_linear_APoT_k2(self, device):
        # weight: fp tensor
        weight = 1000 * torch.rand(5, 3, device=device)

        # activation: fp32 tensor with ~ integer values
        # note: transpose of activation matrix will have dimension (3, 5)
        activation = torch.randint(
            low=0, high=255, size=(5, 3), dtype=torch.float, device=device
        )

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 2)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False, device=device)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.to(dtype=torch.float)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertEqual(apot_linear_result, fp_linear_result, rtol=0, atol=0)


instantiate_device_type_tests(TestNonUniformObserver, globals())


if __name__ == "__main__":
    run_tests()
