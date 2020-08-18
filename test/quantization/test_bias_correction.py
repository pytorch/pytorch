import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import QuantizationTestCase

from torch.quantization import default_qconfig
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper
import torch.quantization._numeric_suite as ns

# import torch.quantization._correct_bias as correct_bias
from torch.quantization._correct_bias import (
    _supported_modules,
    _supported_modules_quantized,
    bias_correction,
    get_module,
    get_param,
    parent_child_names
)

from torchvision.models.quantization import mobilenet_v2
import copy


class TestBiasCorrection(QuantizationTestCase):
    def compute_sqnr(self, x, y):
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn)

    def correct_artificial_bias_float(self, float_model, img_data):
        ''' Adding artificial bias and testing if bias persists after bias
            correction. This test case changes the bias of a floating point submodule
        '''
        artificial_model = copy.deepcopy(float_model)

        # manually changing bias
        for name, submodule in artificial_model.named_modules():
            if type(submodule) in _supported_modules:
                x = get_param(submodule, 'bias')
                if x is not None:
                    x.data = x.data * 3

        bias_correction(float_model, artificial_model, img_data, white_list=_supported_modules)

        for name, submodule in artificial_model.named_modules():
            if isinstance(submodule, ns.Shadow):
                parent_name, child_name = parent_child_names(name)
                parent = get_module(artificial_model, parent_name)
                parent._modules[child_name] = submodule.orig_module

        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in _supported_modules:
                submodule = get_module(float_model, name)
                float_bias = get_param(submodule, 'bias')
                artificial_bias = get_param(artificial_submodule, 'bias')

                self.assertTrue(self.compute_sqnr(float_bias, artificial_bias) > 30,
                                "Correcting quantized bias produced too much noise, sqnr score too low")

    def correct_artificial_bias_quantize(self, float_model, img_data):
        ''' Adding artificial bias and testing if bias persists after bias
            correction. This test case changes the bias of a quantized submodule
        '''
        artificial_model = copy.deepcopy(float_model)
        artificial_model.qconfig = default_qconfig
        torch.quantization.prepare(artificial_model, inplace=True)
        for data in img_data:
            artificial_model(data[0])
        torch.quantization.convert(artificial_model, inplace=True)

        # manually changing bias
        for name, submodule in artificial_model.named_modules():
            if type(submodule) in _supported_modules:
                x = get_param(submodule, 'bias')
                weight = get_param(submodule, 'weight')
                if x is not None:
                    submodule.set_weight_bias(weight, x.data * 3)

        bias_correction(float_model, artificial_model, img_data, white_list=_supported_modules_quantized)

        # Trims off the shadow module,
        for name, submodule in artificial_model.named_modules():
            if isinstance(submodule, ns.Shadow):
                parent_name, child_name = parent_child_names(name)
                parent = get_module(artificial_model, parent_name)
                parent._modules[child_name] = submodule.orig_module

        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in _supported_modules_quantized:
                submodule = get_module(float_model, name)
                float_bias = get_param(submodule, 'bias')
                artificial_bias = get_param(artificial_submodule, 'bias')

                self.assertTrue(self.compute_sqnr(float_bias, artificial_bias) > 30,
                                "Correcting quantized bias produced too much noise, sqnr score too low")

    # def test_pen_paper_1(self):
    #     ''' Testing bias correction on a single Linear module, keeping the weights
    #     constant, but manually changing the bias in the quantized module and verifying
    #     with simple input data that the bias is being corrected
    #     After manual bias change, but before bias correction:
    #         Float module:       [1,1,1] -> [4,4,4,4]
    #         Quantized module    [1,1,1] -> [6,6,6,6]

    #     Expected after bias correction:
    #         Float module:       [1,1,1] -> [4,4,4,4]
    #         Quantized module    [1,1,1] -> [4,4,4,4]
    #     '''
    #     # Linear module that is filled with ones, makes math easier
    #     float_model = nn.Linear(3,4)
    #     float_model.weight.data = torch.ones(float_model.weight.size())
    #     float_model.bias.data = torch.ones(float_model.bias.size())
    #     float_model = QuantWrapper(float_model)

    #     # Quantized module with bias manually changed
    #     artificial_model = copy.deepcopy(float_model)
    #     artificial_model.qconfig = default_qconfig
    #     torch.quantization.prepare(artificial_model, inplace=True)
    #     torch.quantization.convert(artificial_model, inplace=True)
    #     artificial_model.module.bias().data *= 3

    #     # Bias correction
    #     input = [(torch.ones(1,3), 0)]  # single batch with (1,3) tensor
    #     bias_correction(float_model, artificial_model, input)

    #     self.assertTrue(torch.all(float_model(input[0][0]).eq(artificial_model(input[0][0]))))

    # def test_pen_paper_2(self):
    #     ''' Testing bias correction on a single Conv module, keeping the weights
    #     constant, but manually changing the bias in the quantized module and verifying
    #     with simple input data that the bias is being corrected
    #     After manual bias change, but before bias correction:
    #         Float module:       [1,1,1] -> [4,4,4,4]
    #         Quantized module    [1,1,1] -> [13,13,13,13]

    #     Expected after bias correction:
    #         Float module:       [1,1,1] -> [4,4,4,4]
    #         Quantized module    [1,1,1] -> [4,4,4,4]
    #     '''
    #     # Conv module that is filled with ones, makes math easier
    #     float_model = nn.Conv2d(3,4,2,2)
    #     float_model.weight.data = torch.ones(float_model.weight.size())
    #     float_model.bias.data = torch.ones(float_model.bias.size())
    #     float_model = QuantWrapper(float_model)

    #     # Quantized module with bias manually changed
    #     artificial_model = copy.deepcopy(float_model)
    #     artificial_model.qconfig = default_qconfig
    #     torch.quantization.prepare(artificial_model, inplace=True)
    #     torch.quantization.convert(artificial_model, inplace=True)
    #     artificial_model.module.bias().data *= 3

    #     # Bias correction
    #     input = [(torch.ones(1,3,2,2), 0)]  # single batch with (1,3) tensor
    #     bias_correction(float_model, artificial_model, input)

    #     self.assertTrue(torch.all(float_model(input[0][0]).eq(artificial_model(input[0][0]))))

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
        float_model = QuantWrapper(LinearChain())
        img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.correct_artificial_bias_float(float_model, img_data)
        self.correct_artificial_bias_quantize(float_model, img_data)

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
        float_model = QuantWrapper(ConvChain())
        img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.correct_artificial_bias_float(float_model, img_data)
        self.correct_artificial_bias_quantize(float_model, img_data)
