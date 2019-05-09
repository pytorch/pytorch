import torch
import unittest

from test_pytorch_onnx_caffe2 import run_model_test
from test_pytorch_onnx_caffe2 import BATCH_SIZE


class TestCaffe2Backend_opset10(unittest.TestCase):
    embed_params = False

    def test_opset_fallback(self):
        class IsNanModel(torch.nn.Module):
            def forward(self, input):
                return torch.isnan(input)
        x = torch.tensor([1.0, float('nan'), 2.0])
        run_model_test(self, IsNanModel(), train=False, input=x,
                       batch_size=BATCH_SIZE, use_gpu=False, opset_version=10)

    def test_topk(self):
        class TopKModel(torch.nn.Module):
            def forward(self, input):
                return torch.topk(input, 3)
        x = torch.arange(1., 6.)
        run_model_test(self, TopKModel(), train=False, input=x,
                       batch_size=BATCH_SIZE, opset_version=10)
