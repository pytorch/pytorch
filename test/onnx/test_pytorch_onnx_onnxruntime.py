from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import sys
import onnxruntime  # noqa

import torch
import io
import numpy as np
import pytest

class TestONNXRuntime(unittest.TestCase):


    def test_onnxruntime_installed(self):
        self.assertTrue('onnxruntime' in sys.modules)


def verify_with_ort(model, inputs, opset_version):
    pt_outputs = model(*inputs)        
    f = io.BytesIO()
    torch.onnx._export(model, inputs, f, opset_version=opset_version)
    ort_sess = onnxruntime.InferenceSession(f.getvalue())
    ort_input_dict = {ort_input.name: data_input.numpy() for ort_input, data_input in zip(ort_sess.get_inputs(), inputs)}
    ort_outputs = ort_sess.run(None, ort_input_dict)
    assert len(ort_outputs) == len(pt_outputs)
    for i in range(len(ort_outputs)):
        assert np.allclose(ort_outputs[i], pt_outputs[i])

opset_version_config = (9, 10)
desending_config = (True, False)

@pytest.mark.parametrize("opset_version", opset_version_config)
@pytest.mark.parametrize("decending", desending_config)
def test_sort(opset_version, decending):
    if not decending:
        pytest.skip('Test is skipped for ascending sort')

    class SortModel(torch.nn.Module):
        def __init__(self, dim, descending):
            super(SortModel, self).__init__()
            self.dim = dim
            self.descending = descending

        def forward(self, x):
            return torch.sort(x, dim=self.dim, descending=self.descending)

    dim = 1
    model = SortModel(dim, decending)
    x = torch.randn(3, 4)
    verify_with_ort(model, (x,), opset_version)

if __name__ == '__main__':
    unittest.main()
