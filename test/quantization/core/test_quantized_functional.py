# Owner(s): ["oncall: quantization"]

# Torch
import torch
import torch.nn.functional as F
import torch.nn.quantized.functional as qF

# Standard library
import inspect
import numpy as np

# Testing utils
from hypothesis import assume, given
from hypothesis import strategies as st
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    _make_conv_test_input,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import (
    IS_PPC,
    TEST_WITH_UBSAN,
)


class TestQuantizedFunctionalOps(QuantizationTestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.relu(qX)
        qY_hat = F.relu(qX)
        self.assertEqual(qY, qY_hat)

    def _test_conv_api_impl(
        self, qconv_fn, conv_fn, batch_size, in_channels_per_group,
        input_feature_map_size, out_channels_per_group, groups, kernel_size,
        stride, padding, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
        Y_scale, Y_zero_point, use_bias, use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale,
            X_zero_point, W_scale, W_zero_point, use_bias, use_channelwise)

        Y_exp = conv_fn(X, W, b, stride, padding, dilation, groups)
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
        Y_act = qconv_fn(
            X_q, W_q, b, stride, padding, dilation, groups,
            scale=Y_scale, zero_point=Y_zero_point)

        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)
        # We use decimal = 0 to ignore off-by-1 differences between reference
        # and test. Off-by-1 differences arise due to the order of round and
        # zero_point addition operation, i.e., if addition followed by round is
        # used by reference and round followed by addition is used by test, the
        # results may differ by 1.
        # For example, the result of round(2.5) + 1 is 3 while round(2.5 + 1) is
        # 4 assuming the rounding mode is round-to-nearest, ties-to-even.
        np.testing.assert_array_almost_equal(
            Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           L=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv1d_api(
        self, batch_size, in_channels_per_group, L, out_channels_per_group,
        groups, kernel, stride, pad, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv1d function.
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        input_feature_map_size = (L, )
        kernel_size = (kernel, )
        stride = (stride, )
        padding = (pad, )
        dilation = (dilation, )

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv1d
            conv_fn = F.conv1d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv2d_api(
        self, batch_size, in_channels_per_group, H, W, out_channels_per_group,
        groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv2d function.

        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return

        input_feature_map_size = (H, W)
        kernel_size = (kernel_h, kernel_w)
        stride = (stride_h, stride_w)
        padding = (pad_h, pad_w)
        dilation = (dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv2d
            conv_fn = F.conv2d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_d=st.integers(1, 4),
           kernel_h=st.integers(1, 4),
           kernel_w=st.integers(1, 4),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("fbgemm",)))
    def test_conv3d_api(
        self, batch_size, in_channels_per_group, D, H, W,
        out_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation, X_scale,
        X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
        use_channelwise, qengine,
    ):
        # Tests the correctness of the conv3d function.
        # Currently conv3d only supports FbGemm engine

        if qengine not in torch.backends.quantized.supported_engines:
            return

        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv3d
            conv_fn = F.conv3d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(N=st.integers(1, 10),
           C=st.integers(1, 10),
           H=st.integers(4, 8),
           H_out=st.integers(4, 8),
           W=st.integers(4, 8),
           W_out=st.integers(4, 8),
           scale=st.floats(.1, 2),
           zero_point=st.integers(0, 4))
    def test_grid_sample(self, N, C, H, H_out, W, W_out, scale, zero_point):
        X = torch.rand(N, C, H, W)
        X_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        grid = torch.rand(N, H_out, W_out, 2)

        out = F.grid_sample(X_q, grid)
        out_exp = torch.quantize_per_tensor(F.grid_sample(X, grid), scale=scale, zero_point=zero_point, dtype=torch.quint8)
        np.testing.assert_array_almost_equal(
            out.int_repr().numpy(), out_exp.int_repr().numpy(), decimal=0)


class TestSignatureEquivalence(TestCase):
    def _test_same_signature(self, fp_func, q_func,
                             fp_extras=None,
                             q_extras=None,
                             pass_if_no_check=True):
        r"""Checks if two methods, fp_func and q_func have the same signatures.

        TODO: Python 3.10 introduces `inspect.get_annotations()`, which could be used
              https://docs.python.org/3/library/inspect.html#inspect.get_annotations

        The check is done as follows:

        1. Try running `inspect.getfullargspec` on the function to get all the
           args and kwargs.
        2. If (1) fails, try extracting the information from the docstring.
        3. If (2) fails, fail the test if `pass_if_no_checks` is False.
        4. Compare the arguments, while ignoring the fp_extras and q_extras.

        The assumption in (2) is that the docstring's 1st line is always the
        function signature.

        The extra arguments are assumed to come AFTER the common argument.
        The reason for this assumption is that common arguments should be
        accessible positionally, so should come in the same order in both
        functions.

        Args:
            fp_func: Callable FP function
            q_func: Callable quantized function, that is expected to have the same
                    signature as FP.
            fp_extras: Arguments that exist in the FP function, but are not expected
                       in the quantized version
            q_extras: Arguments that exist in the quantized function, but are not
                      needed in the FP version
            pass_if_no_check: Marks test as passed, if there was no way to check for
                              arguments. This might happen if a function is compiled
                              and no information about the signature could be
                              extracted.
        """
        def _get_signature(func):
            if inspect.isbuiltin(func):
                # Cannot get signature of built-ins, bound built-in methods,
                # or any other compiled functions.
                docstring = getattr(func, '__doc__', None)
                if not docstring:
                    return None
                docstring = docstring.splitlines()
                # Find the right line, ideally should be the first one
                sig = None
                for line in docstring:
                    if line.startswith(func.__name__):
                        sig = line
                        break
                if not sig:
                    return None
                signature = inspect._signature_fromstr(
                    cls=inspect.Signature,
                    obj=func, s=sig)
                # Return annotations
                if len(sig.split('->')) > 1:
                    return_annotation = sig.split('->')[1].strip()
                    signature = signature.replace(return_annotation=return_annotation)
                return signature
            return inspect.Signature.from_callable(func)

        fp_sig = _get_signature(fp_func)
        q_sig = _get_signature(q_func)
        if not pass_if_no_check:
            self.assertIsNotNone(fp_sig, msg=f'Cannot get signature of the {fp_func.__name__}')
            self.assertIsNotNone(q_sig, msg=f'Cannot get signature of the {q_func.__name__}')

        if fp_sig is None or q_sig is None:
            # Couldn't extract signature and pass_if_no_check is set
            # There is now
            return

        if not fp_extras and not q_extras:
            # No extras, assume the signatures are identical
            self.assertEqual(fp_sig, q_sig)
        else:
            fp_extras = fp_extras or []
            q_extras = q_extras or []
            fp_args = list(fp_sig.parameters.keys())
            q_args = list(q_sig.parameters.keys())

            # Check if number of common arguments is the same
            num_common_fp_args = len(fp_args) - len(fp_extras)
            num_common_q_args = len(q_args) - len(q_extras)
            self.assertEqual(
                num_common_fp_args, num_common_q_args,
                msg=f'Number of arguments in {fp_func.__name__} and {q_func.__name__}'
                ' don\'t match ({num_common_fp_args} vs. {num_common_q_args})')
            # Check if argument names are same
            # Assume that the extra arguments come AFTER the common ones.
            for fp_arg, q_arg in zip(fp_args[:num_common_fp_args],
                                     q_args[:num_common_fp_args]):
                self.assertEqual(
                    fp_arg, q_arg,
                    msg=f'Argument mismatch: {fp_arg} vs. {q_arg}')

    def test_conv1d(self):
        fp_func = torch.nn.functional.conv1d
        q_func = torch.nn.quantized.functional.conv1d
        self._test_same_signature(fp_func, q_func,
                                  fp_extras=None,
                                  q_extras=['scale', 'zero_point'],
                                  pass_if_no_check=True)

    def test_conv2d(self):
        fp_func = torch.nn.functional.conv2d
        q_func = torch.nn.quantized.functional.conv2d
        self._test_same_signature(fp_func, q_func,
                                  fp_extras=None,
                                  q_extras=['scale', 'zero_point'],
                                  pass_if_no_check=True)

    def test_conv3d(self):
        fp_func = torch.nn.functional.conv3d
        q_func = torch.nn.quantized.functional.conv3d
        self._test_same_signature(fp_func, q_func,
                                  fp_extras=None,
                                  q_extras=['scale', 'zero_point'],
                                  pass_if_no_check=True)
