# Some standard imports
import numpy as np
import io
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
import unittest
import tempfile, shutil

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)


class TestPytorchExportModes(unittest.TestCase):

    def test_protobuf(self):
        torch_model = SuperResolutionNet(upscale_factor=3)
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.PROTOBUF_FILE)

    def test_zipfile(self):
        torch_model = SuperResolutionNet(upscale_factor=3)
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.ZIP_ARCHIVE)

    def test_compressed_zipfile(self):
        torch_model = SuperResolutionNet(upscale_factor=3)
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.COMPRESSED_ZIP_ARCHIVE)

    def test_directory(self):
        torch_model = SuperResolutionNet(upscale_factor=3)
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        d = tempfile.mkdtemp()
        torch.onnx._export(torch_model, (fake_input), d, verbose=False,
                           export_type=torch.onnx.ExportTypes.DIRECTORY)
        shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
