from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnxruntime  # noqa
import torch
import numpy as np
import io

from test_pytorch_common import skipIfUnsupportedMinOpsetVersion


class TestONNXRuntime(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version

    def run_test(self, model, inputs, rtol=1e-05, atol=1e-08):
        outputs = model(inputs)

        # export the model to ONNX
        f = io.BytesIO()
        torch.onnx.export(model, inputs, f,
                          opset_version=self.opset_version,
                          example_outputs=outputs)

        def get_numpy_value_at_index(t, i):
            return t[i].detach().numpy() if t[i].requires_grad else t[i].numpy()

        def get_numpy_value(t):
            return t.detach().numpy() if t.requires_grad else t.numpy()

        def get_ort_inputs():
            ort_inputs = {}
            if isinstance(inputs, torch.Tensor):
                ort_inputs = {ort_sess.get_inputs()[0].name: get_numpy_value(inputs)}
            else:
                for i in range(0, len(outputs)):
                    ort_inputs[ort_sess.get_inputs()[i].name] = get_numpy_value_at_index(inputs, i)
            return ort_inputs

        # compute onnxruntime output prediction
        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        ort_inputs = get_ort_inputs()
        ort_outs = ort_sess.run(None, ort_inputs)

        # compare onnxruntime and PyTorch results
        assert (isinstance(outputs, torch.Tensor) and len(ort_outs) == 1) or \
            len(outputs) == len(ort_outs), \
            "number of outputs differ"

        if isinstance(outputs, torch.Tensor):
            assert np.allclose(get_numpy_value(outputs), ort_outs[0],
                               rtol=rtol, atol=atol)
        else :
            for i in range(0, len(outputs)):
                assert np.allclose(get_numpy_value_at_index(outputs, i), ort_outs[i],
                                   rtol=rtol, atol=atol)

    def test_full_trace(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    def test_full_script(self):
        class FullModelScripting(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_slice_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1:x.size(0)] 

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    def test_interpolate(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2)
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(MyModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=[1, 1, 0.5, 0.5])
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(MyModel(), x)

    def test_layer_norm(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.run_test(model, x, rtol=1e-05, atol=1e-07)

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(torch.nn.Module):
            def forward(self, input):
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

# opset 10 tests
TestONNXRuntime_opset10 = type(str("TestONNXRuntime_opset10"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=10))

if __name__ == '__main__':
    unittest.main()
