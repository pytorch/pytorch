import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization._correct_bias as _correct_bias
from torch.quantization._correct_bias import _supported_modules
from torchvision.models.quantization import mobilenet_v2
import copy
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

    def spnrOfBiasCorrecting(self, float_model, bias_correction, img_data):
        float_model = copy.deepcopy(float_model)
        float_model.qconfig = default_qconfig
        quantized_model = quantize(float_model, default_eval_fn, img_data, inplace=False)
        quantized_model(img_data[0][0])

        bias_correction(float_model, quantized_model, img_data)

        for name, submodule in float_model.named_modules():
            quantized_submodule = _correct_bias.get_module(quantized_model, name)
            float_weight = _correct_bias.get_param(submodule, 'weight')
            quantized_weight = _correct_bias.get_param(quantized_submodule, 'weight')
            if quantized_submodule in _supported_modules:
                if quantized_weight.is_quantized:
                    quantized_weight = quantized_weight.dequantize()

                self.assertTrue(self.computeSqnr(float_weight, quantized_weight) > 35,
                                "Correcting quantized bias produced too much noise, sqnr score too low")

    def test_linear_chain(self):
        class LinearChain(nn.Module):
            def __init__(self):
                super(LinearChain, self).__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.dequant(x)
                return x
        model = LinearChain()
        img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(5)]
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction, img_data)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction, img_data)

    def test_conv_chain(self):
        class ConvChain(nn.Module):
            def __init__(self):
                super(ConvChain, self).__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                x = self.dequant(x)
                return x
        model = ConvChain()
        img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(5)]
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction, img_data)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction, img_data)

    def test_mobilenet(self):
        model = mobilenet_v2(pretrained=True)
        img_data = [(torch.rand(10, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(5)]
        self.spnrOfBiasCorrecting(model, _correct_bias.sequential_bias_correction, img_data)
        self.spnrOfBiasCorrecting(model, _correct_bias.parallel_bias_correction, img_data)
