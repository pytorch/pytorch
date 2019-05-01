import torch

from test_pytorch_onnx_caffe2 import TestCaffe2Backend 
from test_pytorch_onnx_caffe2 import BATCH_SIZE

class TestCaffe2Backend_opset10(TestCaffe2Backend):

    def test_topk(self):
        class TopKModel(torch.nn.Module):
            def forward(self, input):
                return torch.topk(input, 3)
        model = TopKModel()
        x = torch.arange(1., 6.)
        self.run_model_test(TopKModel(), train=False, input=x,
                            batch_size=BATCH_SIZE, opset_version=10)
