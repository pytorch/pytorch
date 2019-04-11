from __future__ import absolute_import, division, print_function, unicode_literals
from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx

import io


class TestUtilityFuns(TestCase):

    def test_is_in_onnx_export(self):
        test_self = self

        class MyModule(torch.nn.Module):
            def forward(self, x):
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                raise ValueError
                return x + 1

        x = torch.randn(3, 4)
        f = io.BytesIO()
        try:
            torch.onnx.export(MyModule(), x, f)
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())


if __name__ == '__main__':
    run_tests()
