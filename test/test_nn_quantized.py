from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized.functional as F
import torch.nn.quantized as nnq
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
        in_features = 10
        out_features = 20
        batch_size = 5
        W = torch.rand(out_features, in_features).float()
        W_q = torch.quantize_linear(W, 0.1, 4, torch.qint8)
        W_pack = torch.ops.quantized.fbgemm_linear_prepack(W_q)
        X = torch.rand(batch_size, in_features).float()
        X_q = torch.quantize_linear(X, 0.2, 10, torch.quint8)
        B = torch.rand(out_features).float()
        B_q = torch.quantize_linear(B, W_q.q_scale() * X_q.q_scale(), 0, torch.qint32)
        out_scale = 0.5
        out_zero_point = 3
        qlinear = nnq.Linear(in_features, out_features)
        qlinear._packed_weight = W_pack
        qlinear.bias = B_q
        qlinear.out_scale = torch.tensor([out_scale])
        qlinear.out_zero_point = torch.tensor([out_zero_point])
        Z_q = qlinear(X_q)
        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.fbgemm_linear(X_q, W_pack, B_q, out_scale, out_zero_point)
        self.assertEqual(Z_ref, Z_q)

        # Test serialization of quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        self.assertEqual(model_dict['bias'], B_q)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model_dict, f)
            f.seek(0)
            loaded_dict = torch.load(f)
        for key in model_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qlinear = nnq.Linear(in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)

        linear_unpack = torch.ops.quantized.fbgemm_linear_unpack
        self.assertEqual(linear_unpack(qlinear._packed_weight),
                         linear_unpack(loaded_qlinear._packed_weight))
        self.assertEqual(qlinear.bias, loaded_qlinear.bias)
        self.assertEqual(qlinear.out_scale, loaded_qlinear.out_scale)
        self.assertEqual(qlinear.out_zero_point, loaded_qlinear.out_zero_point)
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertTrue(hasattr(qlinear, '_packed_weight'))
        self.assertTrue(hasattr(loaded_qlinear, '_packed_weight'))
        self.assertTrue(hasattr(qlinear, 'weight'))
        self.assertTrue(hasattr(loaded_qlinear, 'weight'))
        self.assertEqual(qlinear.weight, loaded_qlinear.weight)
        self.assertEqual(qlinear.weight, torch.ops.quantized.fbgemm_linear_unpack(qlinear._packed_weight))
        Z_q2 = qlinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        # test serialization of module directly - will add this later
        # with tempfile.NamedTemporaryFile() as f:
        #     torch.save(qLinear, f)
        #     f.seek(0)
        #     loaded = torch.load(f)
        # state = qLinear.__getstate__()
        # compareUnpackedWeight(qLinear._packed_weight, loaded._packed_weight)
        # self.assertEqual(qLinear.bias, loaded.bias)
        # self.assertEqual(qLinear.out_scale, loaded.out_scale)
        # self.assertEqual(qLinear.out_zero_point, loaded.out_zero_point)

    def test_quant_dequant_api(self):
        r = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float)
        scale, zero_point, dtype = 1.0, 2, torch.qint8
        # testing Quantize API
        qr = torch.quantize_linear(r, scale, zero_point, dtype)
        quant_m = nnq.Quantize(scale, zero_point, dtype)
        qr2 = quant_m(r)
        self.assertEqual(qr, qr2)
        # testing Dequantize API
        rqr = qr.dequantize()
        dequant_m = nnq.DeQuantize()
        rqr2 = dequant_m(qr2)
        self.assertEqual(rqr, rqr2)



if __name__ == '__main__':
    run_tests()
