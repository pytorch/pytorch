import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import QuantizationTestCase
import torch.quantization._correct_bias as _correct_bias
from torchvision.models.quantization import mobilenet_v2
# import copy
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)

class TestBiasCorrection(QuantizationTestCase):
    def computeSqnr(self, x, y):
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn)

    def spnrOfBiasCorrecting(self, float_model, bias_correction):
        float_model.qconfig = torch.quantization.default_qconfig
        img_data = [(torch.rand(10, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(5)]
        quantized_model = quantize(float_model, default_eval_fn, img_data, inplace=False)

        bias_correction(float_model, quantized_model, img_data)

        for name, submodule in quantized_model.named_modules():
            float_submodule = _correct_bias.get_module(float_model, name)
            float_weight = _correct_bias.get_param(float_submodule, 'weight')
            quantized_weight = _correct_bias.get_param(submodule, 'weight')
            if quantized_weight.is_quantized:
                quantized_weight = quantized_weight.dequantize()

            self.assertTrue(self.computerSqnr(float_weight, quantized_weight) < 35, \
            "Correcting quantized bias produced too much noise, sqnr score too low")

    def test_linear_chain(self):
        class LinearChain(nn.Module):
            def __init__(self):
                super(LinearChain, self).__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x
        model = LinearChain()
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction)

    def test_conv_chain(self):
        class ConvChain(nn.Module):
            def __init__(self):
                super(ConvChain, self).__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x
        model = ConvChain()
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction)

    def test_mobilenet(self):
        model = mobilenet_v2(pretrained=True)
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction)
