import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import QuantizationTestCase

from torch.quantization import default_qconfig
from torch.quantization import QuantStub, DeQuantStub

import torch.quantization._correct_bias as correct_bias
from torch.quantization._correct_bias import _supported_modules, _supported_modules_quantized

# from torchvision.models.quantization import mobilenet_v2
import copy


class TestBiasCorrection(QuantizationTestCase):
    def compute_sqnr(self, x, y):
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn)

    def correct_artificial_bias(self, float_model, bias_correction, img_data):
        ''' Adding artificial bias and testing if bias persists after bias
            correction
        '''
        artificial_model = copy.deepcopy(float_model)
        artificial_model.qconfig = default_qconfig
        torch.quantization.prepare(artificial_model, inplace=True)
        artificial_model(img_data[0][0])
        torch.quantization.convert(artificial_model, inplace=True)

        # manually changing bias
        for name, submodule in artificial_model.named_modules():
            if type(submodule) in _supported_modules_quantized:
                x = correct_bias.get_param(submodule, 'bias')
                if x is not None:
                    x.data = x.data * 10

        bias_correction(float_model, artificial_model, img_data)

        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in correct_bias._supported_modules_quantized:
                submodule = correct_bias.get_module(float_model, name)
                float_bias = correct_bias.get_param(submodule, 'bias')
                artificial_bias = correct_bias.get_param(artificial_submodule, 'bias')

                if artificial_submodule in _supported_modules_q:
                    # if artificial_bias.is_quantized:  # jerry says this shouldn't be needed, but artificial model is quantized
                    #     artificial_bias = artificial_bias.dequantize()

                    # big ruh roh, this wasn't executing the entire time
                    self.assertTrue(self.computeSqnr(float_bias, artificial_bias) > 35,
                                    "Correcting quantized bias produced too much noise, sqnr score too low")

    # abuse 4 bit quantization, try accuracy on one layer or something?
    # hard code some expected numbers
    def four_bit_test(self, float_model, bias_correction, img_data):
        pass

    def pen_paper_test(self, float_model, bias_correction, img_data):
        pass

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
        float_model = LinearChain()
        img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.correct_artificial_bias(float_model, correct_bias.bias_correction, img_data)

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
        float_model = ConvChain()
        img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.correct_artificial_bias(float_model, correct_bias.bias_correction, img_data)

    def test_mobilenet(self):
        float_model = mobilenet_v2(pretrained=True, quantize=False)
        float_model.eval()
        float_model.fuse_model()
        img_data = [(torch.rand(10, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.correct_artificial_bias(float_model, correct_bias.bias_correction, img_data)
