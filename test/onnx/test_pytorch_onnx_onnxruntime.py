from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import sys
import onnxruntime  # noqa

import torch

class TestONNXRuntime(unittest.TestCase):

    def test_onnxruntime_installed(self):
        self.assertTrue('onnxruntime' in sys.modules)

    def test_sort(self):
        class SortModel(torch.nn.Module):
            def __init__(self, dim, descending):
                super(SortModel, self).__init__()
                self.dim = dim
                self.descending = descending

            def forward(self, x):
                return torch.sort(x, dim=self.dim, descending=self.descending)

        dim = 1
        decending = False
        model = SortModel(dim, decending)

        x = torch.randn(3, 4)
        pt_sorted, pt_indices = model(x)

        # f = io.BytesIO()
        # torch.onnx._export(model, (x,), f)
        # ort_sess = onnxruntime.InferenceSession(f.getvalue())
        # ort_sorted, ort_indices = ort_sess.run(None, {ort_sess.get_inputs()[0].name: x.numpy()})

        # np.allclose(ort_sorted, pt_sorted)
        # np.allclose(ort_indices, pt_indices)

if __name__ == '__main__':
    unittest.main()
