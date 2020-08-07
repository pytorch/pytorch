import torch
import torch.nn as nn
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization._adaround import _supported_modules
from torchvision.models.quantization import mobilenet_v2
import copy
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)

class TestAdaround(QuantizationTestCase):
    def single_layer_adaround(self, model, adaround_func, img_data):
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
        self.single_layer_adaround(float_model, _adaround.learn_adaround, img_data)

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
        self.single_layer_adaround(float_model, _adaround.learn_adaround, img_data)

    def test_mobilenet(self):
        float_model = mobilenet_v2(pretrained=True, quantize=False)
        float_model.eval()
        float_model.fuse_model()
        img_data = [(torch.rand(10, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                    for _ in range(50)]
        self.single_layer_adaround(float_model, _adaround.learn_adaround, img_data)
