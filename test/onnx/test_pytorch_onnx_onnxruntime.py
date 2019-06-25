from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnxruntime  # noqa
import torch
import numpy as np
import io

class TestONNXRuntime(unittest.TestCase):

    def run_test(self, model, input, output, opset_version=None, example_outputs=None):
        # export the model to ONNX
        f = io.BytesIO()
        torch.onnx.export(model, input, f,
                          opset_version=opset_version,
                          example_outputs=example_outputs)

        # compute onnxruntime output prediction
        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        ort_outs = ort_sess.run(None, {ort_sess.get_inputs()[0].name: input.numpy()})

        # compare onnxruntime and PyTorch results 
        if output.requires_grad:
            output = output.detach().numpy()
        else:
            output = output.numpy()
        np.allclose(output, ort_outs[0])


    def test_layer_norm(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        output = model(input)

        self.run_test(model, x)


if __name__ == '__main__':
    unittest.main()
