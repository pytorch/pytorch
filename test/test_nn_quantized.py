from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional as qF
from torch.nn.quantized.modules import Conv2d
from torch.nn.quantized.modules.conv import _conv_output_shape

import numpy as np
from common_utils import TestCase, run_tests

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis_utils import qtensors_conv

def _quantize(x, scale, zero_point, qmin=0, qmax=255):
    """Quantizes a numpy array."""
    qx = np.round(x / scale + zero_point)
    qx = np.clip(qx, qmin, qmax).astype(np.uint8)
    return qx

class ModuleAPITest(TestCase):
    def test_functional_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        Y = X.numpy().copy()
        Y[Y < 0] = 0
        qY = _quantize(Y, scale, zero_point)
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY_hat = qF.relu(qX)
        np.testing.assert_equal(qY, qY_hat.int_repr())

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
        assume(iC % groups == 0)
        # assume(iC // groups > 0)
        # assume(oC % groups == 0)
        # assume(oC // groups > 0)
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
        conv_2d = Conv2d(weight=q_filters, bias=q_bias,
                         scale=scale, zero_point=zero_point,
                         dtype=torch_type,
                         stride=stride, padding=i_padding,
                         dilation=dilation, groups=groups,
                         padding_mode='zeros')
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
