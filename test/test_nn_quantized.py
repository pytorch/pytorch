from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized.functional as F
import torch.nn.quantized as nnq
import numpy as np
from common_utils import TestCase, run_tests, tempfile

'''
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `caffe2/test/test_quantized.py`.
'''

class FunctionalAPITest(TestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.ops.quantized.relu(qX)
        qY_hat = F.relu(qX)
        self.assertEqual(qY, qY_hat)


class ModuleAPITest(TestCase):
    def test_linear_api(self):
        """test API functionality for nn.quantized.linear"""
        input_channels = 10
        output_channels = 20
        batch_size = 5
        W = torch.rand(output_channels, input_channels).float()
        W_q = torch.quantize_linear(W, 0.1, 4, torch.qint8)
        X = torch.rand(batch_size, input_channels).float()
        X_q = torch.quantize_linear(X, 0.2, 10, torch.quint8)
        B = torch.rand(output_channels).float()
        B_q = torch.quantize_linear(B, W_q.q_scale() * X_q.q_scale(), 0, torch.qint32)
        out_scale = 0.5
        out_zero_point = 3
        qLinear = nnq.Linear(output_channels, input_channels)
        # Later diff replaces this with set_state/get_state
        qLinear._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(W_q)
        qLinear.bias = B_q
        qLinear.output_scale = torch.Tensor([out_scale])
        qLinear.output_zero_point = torch.Tensor([out_zero_point])
        Z_q = qLinear(X_q)
        # Check if the module implementation matches calling the
        # ops directly
        W_pack = torch.ops.quantized.fbgemm_linear_prepack(W_q)
        Z_ref = torch.ops.quantized.fbgemm_linear(X_q, W_pack, B_q, out_scale, out_zero_point)
        self.assertEqual(Z_ref, Z_q)

        # Test serialization of quantized Linear Module using state_dict
        model_dict = qLinear.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model_dict, f)
            f.seek(0)
            loaded_dict = torch.load(f)
        for key in model_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        qLinear2 = nnq.Linear(output_channels, input_channels)
        qLinear2.load_state_dict(loaded_dict)

        def compareUnpackedWeight(w1, w2):
            linear_unpack = torch.ops.quantized.fbgemm_linear_unpack
            self.assertEqual(linear_unpack(w1), linear_unpack(w2))
        compareUnpackedWeight(qLinear._packed_weight, qLinear2._packed_weight)
        self.assertEqual(qLinear.bias, qLinear2.bias)
        self.assertEqual(qLinear.output_scale, qLinear2.output_scale)
        self.assertEqual(qLinear.output_zero_point, qLinear2.output_zero_point)
        self.assertTrue(dir(qLinear) == dir(qLinear))
        self.assertFalse('weight' in qLinear.__dict__)
        self.assertFalse('weight' in qLinear2.__dict__)
        Z_q2 = qLinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        # test serialization of module directly - will add this later
        # with tempfile.NamedTemporaryFile() as f:
        #     torch.save(qLinear, f)
        #     f.seek(0)
        #     loaded = torch.load(f)
        # state = qLinear.__getstate__()
        # compareUnpackedWeight(qLinear._packed_weight, loaded._packed_weight)
        # self.assertEqual(qLinear.bias, loaded.bias)
        # self.assertEqual(qLinear.output_scale, loaded.output_scale)
        # self.assertEqual(qLinear.output_zero_point, loaded.output_zero_point)


if __name__ == '__main__':
    run_tests()
