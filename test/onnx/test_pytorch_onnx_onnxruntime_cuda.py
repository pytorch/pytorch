import unittest
import onnxruntime  # noqa
import torch

from test_pytorch_common import skipIfUnsupportedMinOpsetVersion
from test_pytorch_common import skipIfNoCuda

from test_pytorch_onnx_onnxruntime import TestONNXRuntime

class TestONNXRuntime_cuda(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version
    keep_initializers_as_inputs = True
    use_new_jit_passes = True
    onnx_shape_inference = True

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    def test_gelu_fp16(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        x = torch.randn(2, 4, 5, 6, requires_grad=True, dtype=torch.float16, device=torch.device('cuda'))
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

TestONNXRuntime_cuda.setUp = TestONNXRuntime.setUp
TestONNXRuntime_cuda.run_test = TestONNXRuntime.run_test

if __name__ == '__main__':
    unittest.main(TestONNXRuntime_cuda())
