from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.quantized.functional as qF

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis_utils import qtensors_conv

from common_utils import TestCase, run_tests


class FunctionalAPITest(TestCase):
    """Computes the output shape given convolution parameters."""
    def _conv_output_shape(self, input_size, kernel_size, padding, stride,
                           dilation):
        return np.floor((input_size + 2 * padding - kernel_size
                         - (kernel_size - 1) * (dilation - 1)) / stride) + 1

    @given(Q=qtensors_conv(min_batch=1, max_batch=3,
                           min_in_channels=1, max_in_channels=7,
                           min_out_channels=1, max_out_channels=7,
                           H_range=(6, 12), W_range=(6, 12),
                           kH_range=(3, 5), kW_range=(3, 5),
                           dtypes=((torch.quint8, np.uint8, 0),),
                           max_groups=4),
           padH=st.integers(1, 3), padW=st.integers(1, 3),
           sH=st.integers(1, 3), sW=st.integers(1, 3),
           dH=st.integers(1, 2), dW=st.integers(1, 2),
           prepacked=st.booleans())
    def test_conv_api(self, Q, padH, padW, sH, sW, dH, dW, prepacked):
        """Tests the correctness of the conv functional.

        The correctness is defined by the behavior being similar to the
        `quantized._ops` implementation.
        """
        # Random iunputs
        X, (scale, zero_point), (qmin, qmax), (torch_type, np_type) = Q
        (inputs, filters, bias, groups) = X

        iC, oC = inputs.shape[1], filters.shape[0]

        iH, iW = inputs.shape[2:]
        kH, kW = filters.shape[2:]
        assume(kH // 2 >= padH)
        assume(kW // 2 >= padW)
        oH = self._conv_output_shape(iH, kH, padH, sH, dH)
        assume(oH > 0)
        oW = self._conv_output_shape(iW, kW, padW, sW, dW)
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

        q_inputs = torch.quantize_linear(i_NHWC, scale, zero_point, torch.quint8)
        q_filters = torch.quantize_linear(w_RSCK, scale, zero_point, torch.qint8)
        q_filters_ref = torch.ops.quantized.fbgemm_conv_prepack(q_filters,
                                                                groups)
        q_bias = torch.quantize_linear(bias, scale, zero_point, torch.qint32)

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
