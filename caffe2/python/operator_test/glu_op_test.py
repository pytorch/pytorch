from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import assume, given
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestGlu(hu.HypothesisTestCase):
    @given(
        X=hu.tensor(),
        **hu.gcs
    )
    def test_glu(self, X, gc, dc):

        def glu_ref(X):
            ndim = X.ndim
            M = 1
            for i in range(ndim - 1):
                M *= X.shape[i]
            N = X.shape[ndim - 1]
            N2 = int(N / 2)
            yShape = list(X.shape)
            yShape[ndim - 1] = int(yShape[ndim - 1] / 2)
            Y = np.zeros(yShape)
            for i in range(0, M):
                for j in range(0, N2):
                    x1 = X.flat[i * N + j]
                    x2 = X.flat[i * N + j + N2]
                    Y.flat[i * N2 + j] = x1 * (1. / (1. + np.exp(-x2)))
            return [Y]

        # Test only valid tensors.
        assume(X.shape[X.ndim - 1] % 2 == 0)
        op = core.CreateOperator("Glu", ["X"], ["Y"])
        self.assertReferenceChecks(gc, op, [X], glu_ref)


if __name__ == "__main__":
    unittest.main()
