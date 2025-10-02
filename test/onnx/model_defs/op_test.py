# Owner(s): ["module: onnx"]

import torch
import torch.nn as nn


class DummyNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(3),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

    def forward(self, x):
        output = self.features(x)
        return output.view(-1, 1).squeeze(1)


class ConcatNet(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, 1)


class PermuteNet(nn.Module):
    def forward(self, input):
        return input.permute(2, 3, 0, 1)


class PReluNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.PReLU(3),
        )

    def forward(self, x):
        output = self.features(x)
        return output


class FakeQuantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fake_quant = torch.ao.quantization.FakeQuantize()
        self.fake_quant.disable_observer()

    def forward(self, x):
        output = self.fake_quant(x)
        return output
