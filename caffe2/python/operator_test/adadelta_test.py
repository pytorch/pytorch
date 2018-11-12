from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import hypothesis
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestAdadelta(serial.SerializedTestCase):
    @staticmethod
    def ref_adadelta(param_in,
                     mom_in,
                     mom_delta_in,
                     grad, lr,
                     epsilon,
                     decay,
                     using_fp16=False):
        param_in_f32 = param_in
        mom_in_f32 = mom_in
        mom_delta_in_f32 = mom_delta_in
        if(using_fp16):
            param_in_f32 = param_in.astype(np.float32)
            mom_in_f32 = mom_in.astype(np.float32)
            mom_delta_in_f32 = mom_delta_in.astype(np.float32)

        mom_out = decay * mom_in_f32 + (1.0 - decay) * grad * grad
        new_grad = (np.sqrt(mom_delta_in_f32 + epsilon) /
                    np.sqrt(mom_out + epsilon)) * grad
        param_out = param_in_f32 + lr * new_grad
        mom_delta_out = decay * mom_delta_in_f32 + (1.0 - decay
                                                    ) * new_grad * new_grad
        if(using_fp16):
            return (param_out.astype(np.float16), mom_out.astype(np.float16),
                    mom_delta_out.astype(np.float16))
        else:
            return (param_out.astype(np.float32), mom_out.astype(np.float32),
                    mom_delta_out.astype(np.float32))

    @serial.given(inputs=hu.tensors(n=4),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           decay=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs)
    def test_adadelta(self, inputs, lr, epsilon, decay, gc, dc):
        param, moment, moment_delta, grad = inputs
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adadelta",
            ["param", "moment", "moment_delta", "grad", "lr"],
            ["param", "moment", "moment_delta"],
            epsilon=epsilon,
            decay=decay,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc, op,
            [param, moment, moment_delta, grad, lr],
            functools.partial(self.ref_adadelta, epsilon=epsilon, decay=decay))

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=4),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           decay=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs)
    def test_sparse_adadelta(self, inputs, lr, epsilon, decay, gc, dc):
        param, moment, moment_delta, grad = inputs
        moment = np.abs(moment)
        lr = np.array([lr], dtype=np.float32)

        # Create an indexing array containing values that are lists of indices,
        # which index into grad
        indices = np.random.choice(np.arange(grad.shape[0]),
            size=np.random.randint(grad.shape[0]), replace=False)

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseAdadelta",
            ["param", "moment", "moment_delta", "indices", "grad", "lr"],
            ["param", "moment", "moment_delta"],
            epsilon=epsilon,
            decay=decay,
            device_option=gc)

        def ref_sparse(param, moment, moment_delta, indices, grad, lr, decay,
                       ref_using_fp16):
            param_out = np.copy(param)
            moment_out = np.copy(moment)
            moment_delta_out = np.copy(moment_delta)
            for i, index in enumerate(indices):
                param_out[index], moment_out[index], moment_delta_out[
                    index] = self.ref_adadelta(param[index], moment[index],
                                               moment_delta[index], grad[i], lr,
                                               epsilon, decay, ref_using_fp16)
            return (param_out, moment_out, moment_delta_out)

        ref_using_fp16_values = [False]
        if dc == hu.gpu_do:
            ref_using_fp16_values.append(True)

        for ref_using_fp16 in ref_using_fp16_values:
            moment_i = None
            moment_delta_i = None
            param_i = None
            if(ref_using_fp16):
                moment_i = moment.astype(np.float16)
                moment_delta_i = moment_delta.astype(np.float16)
                param_i = param.astype(np.float16)
            else:
                moment_i = moment.astype(np.float32)
                moment_delta_i = moment_delta.astype(np.float32)
                param_i = param.astype(np.float32)

                self.assertReferenceChecks(gc, op, [
                    param_i, moment_i, moment_delta_i, indices, grad, lr, decay,
                    ref_using_fp16
                ], ref_sparse)

    @serial.given(inputs=hu.tensors(n=3),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           decay=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs)
    def test_sparse_adadelta_empty(self, inputs, lr, epsilon, decay,
                                  data_strategy, gc, dc):
        param, moment, moment_delta = inputs
        moment = np.abs(moment)
        lr = np.array([lr], dtype=np.float32)

        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0,), dtype=np.int64)

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        op = core.CreateOperator(
            "SparseAdadelta",
            ["param", "moment", "moment_delta", "indices", "grad", "lr"],
            ["param", "moment", "moment_delta"],
            epsilon=epsilon,
            decay=decay,
            device_option=gc)

        def ref_sparse_empty(param, moment, moment_delta, indices, grad, lr, decay):
            param_out = np.copy(param)
            moment_out = np.copy(moment)
            moment_delta_out = np.copy(moment_delta)
            return (param_out, moment_out, moment_delta_out)

        ref_using_fp16_values = [False]
        if dc == hu.gpu_do:
            ref_using_fp16_values.append(True)

        for ref_using_fp16 in ref_using_fp16_values:
            moment_i = None
            moment_delta_i = None
            param_i = None
            if(ref_using_fp16):
                moment_i = moment.astype(np.float16)
                moment_delta_i = moment_delta.astype(np.float16)
                param_i = param.astype(np.float16)
            else:
                moment_i = moment.astype(np.float32)
                moment_delta_i = moment_delta.astype(np.float32)
                param_i = param.astype(np.float32)

        self.assertReferenceChecks(
            gc,
            op,
            [param_i, moment_i, moment_delta_i, indices, grad, lr, decay],
            ref_sparse_empty
        )
