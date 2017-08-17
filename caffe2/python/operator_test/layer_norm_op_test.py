from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from functools import reduce
from hypothesis import given
from operator import mul
import caffe2.python.hypothesis_test_util as hu
import numpy as np


class TestLayerNormOp(hu.HypothesisTestCase):
    @given(X=hu.tensors(n=1), **hu.gcs_cpu_only)
    def test_layer_norm_op(self, X, gc, dc):
        X = X[0]
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        axis = np.random.randint(0, len(X.shape))
        epsilon = 0.001
        op = core.CreateOperator(
            "LayerNorm",
            ["input"],
            ["output", "mean", "stdev"],
            axis=axis,
            epsilon=epsilon,
        )

        def layer_norm_ref(X):
            left = reduce(mul, X.shape[:axis], 1)
            reshaped = np.reshape(X, [left, -1])
            mean = np.mean(reshaped, axis=1).reshape([left, 1])
            stdev = np.sqrt(
                np.mean(np.power(reshaped, 2), axis=1).reshape([left, 1]) -
                np.power(mean, 2)
            )
            norm = (reshaped - mean) / (stdev + epsilon)
            norm = np.reshape(norm, X.shape)
            mean = np.reshape(mean, X.shape[:axis] + (1,))
            stdev = np.reshape(stdev, X.shape[:axis] + (1,))
            return [norm, mean, stdev]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=layer_norm_ref
        )
