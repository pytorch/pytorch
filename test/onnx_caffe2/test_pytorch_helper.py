# Owner(s): ["module: onnx"]

# Some standard imports
import unittest

import numpy as np
import pytorch_test_common

import torch.nn.init as init
import torch.onnx
from caffe2.python.core import workspace
from caffe2.python.model_helper import ModelHelper
from pytorch_helper import PyTorchModule
from torch import nn
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


class TestCaffe2Backend(pytorch_test_common.ExportTestCase):
    @skipIfNoLapack
    @unittest.skip("test broken because Lapack was always missing.")
    def test_helper(self):
        class SuperResolutionNet(nn.Module):
            def __init__(self, upscale_factor, inplace=False):
                super().__init__()

                self.relu = nn.ReLU(inplace=inplace)
                self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
                self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
                self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
                self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
                self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

                self._initialize_weights()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.pixel_shuffle(self.conv4(x))
                return x

            def _initialize_weights(self):
                init.orthogonal(self.conv1.weight, init.calculate_gain("relu"))
                init.orthogonal(self.conv2.weight, init.calculate_gain("relu"))
                init.orthogonal(self.conv3.weight, init.calculate_gain("relu"))
                init.orthogonal(self.conv4.weight)

        torch_model = SuperResolutionNet(upscale_factor=3)

        fake_input = torch.randn(1, 1, 224, 224, requires_grad=True)

        # use ModelHelper to create a C2 net
        helper = ModelHelper(name="test_model")
        start = helper.Sigmoid(["the_input"])
        # Embed the ONNX-converted pytorch net inside it
        (toutput,) = PyTorchModule(helper, torch_model, (fake_input,), [start])
        output = helper.Sigmoid(toutput)

        workspace.RunNetOnce(helper.InitProto())
        workspace.FeedBlob("the_input", fake_input.data.numpy())
        # print([ k for k in workspace.blobs ])
        workspace.RunNetOnce(helper.Proto())
        c2_out = workspace.FetchBlob(str(output))

        torch_out = torch.sigmoid(torch_model(torch.sigmoid(fake_input)))

        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)


if __name__ == "__main__":
    common_utils.run_tests()
