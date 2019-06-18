from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import sys
import onnxruntime  # noqa
import torch
import numpy as np
import io

class TestONNXRuntime(unittest.TestCase):

    def test_onnxruntime_installed(self):
        self.assertTrue('onnxruntime' in sys.modules)

    def test_full(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        model = FullModel()
        output = model(x)

        f = io.BytesIO()
        torch.onnx.export(model, x, f)

        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        ort_outs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: x.numpy()})
        np.allclose(output.numpy(), ort_outs[0])

        class FullModelScripting(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        model_scripting = FullModelScripting()
        output_scripting = model_scripting(x)

        f_scripting = io.BytesIO()
        torch.onnx.export(model_scripting, x, f_scripting, example_outputs=output_scripting)

        ort_sess_scripting = onnxruntime.InferenceSession(f_scripting.getvalue())
        ort_outs_scripting = ort_sess_scripting.run(None, {ort_sess_scripting.get_inputs()[0].name: x.numpy()})
        np.allclose(output_scripting.numpy(), ort_outs_scripting[0])


if __name__ == '__main__':
    unittest.main()
