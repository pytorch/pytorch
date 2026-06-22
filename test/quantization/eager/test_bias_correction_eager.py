# Owner(s): ["oncall: quantization"]

import copy

import torch
import torch.ao.ns._numeric_suite as ns
import torch.nn as nn
from torch.ao.quantization import default_qconfig, QuantWrapper
from torch.ao.quantization._correct_bias import (
    _supported_modules,
    _supported_modules_quantized,
    bias_correction,
    get_module,
    get_param,
    parent_child_names,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import raise_on_run_directly


class TestBiasCorrectionEager(QuantizationTestCase):
    def compute_sqnr(self, x, y):
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn)

    def correct_artificial_bias_quantize(self, float_model, img_data):
        """Adding artificial bias and testing if bias persists after bias
        correction. This test case changes the bias of a quantized submodule
        """
        artificial_model = copy.deepcopy(float_model)
        artificial_model.qconfig = default_qconfig
        torch.ao.quantization.prepare(artificial_model, inplace=True)
        for data in img_data:
            artificial_model(data[0])
        torch.ao.quantization.convert(artificial_model, inplace=True)

        # manually changing bias
        for submodule in artificial_model.modules():
            if type(submodule) in _supported_modules:
                x = get_param(submodule, "bias")
                weight = get_param(submodule, "weight")
                if x is not None:
                    submodule.set_weight_bias(weight, x.data * 3)

        bias_correction(
            float_model,
            artificial_model,
            img_data,
            target_modules=_supported_modules_quantized,
        )

        # Trims off the shadow module,
        for name, submodule in artificial_model.named_modules():
            if isinstance(submodule, ns.Shadow):
                parent_name, child_name = parent_child_names(name)
                parent = get_module(artificial_model, parent_name)
                parent._modules[child_name] = submodule.orig_module

        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in _supported_modules_quantized:
                submodule = get_module(float_model, name)
                float_bias = get_param(submodule, "bias")
                artificial_bias = get_param(artificial_submodule, "bias")

                self.assertTrue(
                    self.compute_sqnr(float_bias, artificial_bias) > 30,
                    "Correcting quantized bias produced too much noise, sqnr score too low",
                )

    @skipIfNoFBGEMM
    def test_linear_chain(self):
        class LinearChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        float_model = QuantWrapper(LinearChain())
        img_data = [
            (
                torch.rand(10, 3, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            )
            for _ in range(50)
        ]
        self.correct_artificial_bias_quantize(float_model, img_data)

    @skipIfNoFBGEMM
    def test_conv_chain(self):
        class ConvChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x

        float_model = QuantWrapper(ConvChain())
        img_data = [
            (
                torch.rand(10, 3, 125, 125, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            )
            for _ in range(50)
        ]
        self.correct_artificial_bias_quantize(float_model, img_data)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
