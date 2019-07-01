from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized as nnq
import torch.nn.quantized.functional as F
import torch.nn.quantized.functional as qF
from torch.nn.quantized.modules import Conv2d
from torch.nn.quantized.modules.conv import _conv_output_shape
from common_utils import TestCase, run_tests, tempfile
from common_utils import _quantize

import numpy as np

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis_utils import qtensors_conv

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



    @given(Q=qtensors_conv(min_batch=1, max_batch=3,
                           min_in_channels=1, max_in_channels=5,
                           min_out_channels=1, max_out_channels=5,
                           H_range=(6, 12), W_range=(6, 12),
                           kH_range=(3, 5), kW_range=(3, 5),
                           dtypes=((torch.quint8, np.uint8, 0),),
                           max_groups=4),
           padH=st.integers(1, 3), padW=st.integers(1, 3),
           dH=st.integers(1, 2), dW=st.integers(1, 2),
           sH=st.integers(1, 3), sW=st.integers(1, 3))
    def test_conv_api(self, Q, padH, padW, dH, dW, sH, sW):
        """Tests the correctness of the conv module.

        The correctness is defined against the functional implementation.
        """
        ref_op = qF.conv2d

        # Random iunputs
        X, (scale, zero_point), (qmin, qmax), (torch_type, np_type) = Q
        (inputs, filters, bias, groups) = X

        iC, oC = inputs.shape[1], filters.shape[0]
        iH, iW = inputs.shape[2:]
        kH, kW = filters.shape[2:]
        assume(kH // 2 >= padH)
        assume(kW // 2 >= padW)
        oH = _conv_output_shape(iH, kH, padH, sH, dH)
        assume(oH > 0)
        oW = _conv_output_shape(iW, kW, padW, sW, dW)
        assume(oW > 0)
        inputs = torch.from_numpy(inputs).to(torch.float)
        filters = torch.from_numpy(filters).to(torch.float)
        bias = torch.from_numpy(bias).to(torch.float)
        kernel_size = (kH, kW)
        stride = (sH, sW)
        i_padding = (padH, padW)
        dilation = (dH, dW)

        i_NHWC = inputs.permute([0, 2, 3, 1]).contiguous()
        w_RSCK = filters.permute([0, 2, 3, 1]).contiguous()
        q_inputs = torch.quantize_linear(i_NHWC, scale, zero_point, torch.quint8)
        q_filters = torch.quantize_linear(w_RSCK, scale, zero_point, torch.qint8)
        q_bias = torch.quantize_linear(bias, scale, zero_point, torch.qint32)

        # Results check
        conv_2d = Conv2d(in_channels=iC, out_channels=oC, kernel_size=(kH, kW),
                         stride=stride, padding=i_padding,
                         dilation=dilation, groups=groups,
                         padding_mode='zeros')
        conv_2d.weight = q_filters
        conv_2d.bias = q_bias
        conv_2d.scale = scale
        conv_2d.zero_point = zero_point
        try:
            ref_result = qF.conv2d(q_inputs, q_filters, bias=q_bias,
                                   scale=scale, zero_point=zero_point,
                                   stride=stride, padding=i_padding,
                                   dilation=dilation, groups=groups,
                                   prepacked=False, dtype=torch_type)
        except RuntimeError as e:
            # We should be throwing the same error.
            e_msg = str(e).split("\n")[0].split("(")[0].strip()
            np.testing.assert_raises_regex(type(e), e_msg,
                                           conv_2d, q_inputs)
        else:
            q_result = conv_2d(q_inputs)
            np.testing.assert_equal(ref_result.int_repr().numpy(),
                                    q_result.int_repr().numpy())

if __name__ == '__main__':
    run_tests()
