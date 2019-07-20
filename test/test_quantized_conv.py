from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.quantized.functional as qF

from hypothesis import assume, given
from hypothesis import strategies as st
import hypothesis_utils as hu

from common_quantized import _conv_output_shape
from common_utils import TestCase, run_tests


class FunctionalAPITest(TestCase):
    @given(X=hu.tensor_conv2d(min_batch=1, max_batch=3,
                              min_in_channels=1, max_in_channels=7,
                              min_out_channels=1, max_out_channels=7,
                              H_range=(6, 12), W_range=(6, 12),
                              kH_range=(3, 5), kW_range=(3, 5),
                              max_groups=4,
                              qparams=[hu.qparams(dtypes=torch.quint8,
                                                  zero_point_min=0,
                                                  zero_point_max=0),
                                       hu.qparams(dtypes=torch.qint8,
                                                  zero_point_min=0,
                                                  zero_point_max=0),
                                       hu.qparams(dtypes=torch.qint32,
                                                  zero_point_min=0,
                                                  zero_point_max=0)]),
           padH=st.integers(1, 3), padW=st.integers(1, 3),
           sH=st.integers(1, 3), sW=st.integers(1, 3),
           dH=st.integers(1, 2), dW=st.integers(1, 2),
           prepacked=st.booleans())
    def test_conv_api(self, X, padH, padW, sH, sW, dH, dW, prepacked):
        """Tests the correctness of the conv functional.

        The correctness is defined by the behavior being similar to the
        `quantized._ops` implementation.
        """
        # Random inputs
        # X, (scale, zero_point, torch_type) = X
        (inputs, filters, bias, groups) = X
        inputs, (inputs_scale, inputs_zero_point, inputs_qtype) = inputs
        filters, (filters_scale, filters_zero_point, filters_qtype) = filters
        bias, (bias_scale, bias_zero_point, bias_qtype) = bias

        scale, zero_point = inputs_scale, inputs_zero_point
        torch_type = inputs_qtype

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

        # Quantized inputs
        i_NHWC = inputs.permute([0, 2, 3, 1]).contiguous()
        w_RSCK = filters.permute([0, 2, 3, 1]).contiguous()

        q_inputs = torch.quantize_linear(i_NHWC, inputs_scale, inputs_zero_point,
                                         inputs_qtype)
        q_filters = torch.quantize_linear(w_RSCK, filters_scale,
                                          filters_zero_point, filters_qtype)
        q_filters_ref = torch.ops.quantized.fbgemm_conv_prepack(q_filters,
                                                                groups)
        q_bias = torch.quantize_linear(bias, bias_scale, bias_zero_point,
                                       bias_qtype)

        # Reference op
        ref_op = torch.ops.quantized.fbgemm_conv2d

        # Results check
        try:
            ref_result = ref_op(q_inputs, q_filters_ref, q_bias, stride,
                                i_padding, dilation,
                                groups, scale, zero_point)
        except RuntimeError as e:
            e_msg = str(e).split("\n")[0].split("(")[0].strip()
            np.testing.assert_raises_regex(
                type(e), e_msg, qF.conv2d,
                q_inputs, q_filters_ref, bias=q_bias,
                scale=scale, zero_point=zero_point,
                stride=stride, padding=i_padding, dilation=dilation,
                groups=groups, prepacked=True, dtype=torch_type)
        else:
            if prepacked:
                q_filters = torch.ops.quantized.fbgemm_conv_prepack(q_filters,
                                                                    groups)
            q_result = qF.conv2d(q_inputs, q_filters, bias=q_bias,
                                 scale=scale, zero_point=zero_point,
                                 stride=stride, padding=i_padding,
                                 dilation=dilation, groups=groups,
                                 prepacked=prepacked, dtype=torch_type)

            np.testing.assert_equal(ref_result.int_repr().numpy(),
                                    q_result.int_repr().numpy())

if __name__ == "__main__":
    run_tests()
