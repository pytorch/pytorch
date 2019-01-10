from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dyndep
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/torch:th_ops')


try:
    dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/torch:th_ops_gpu')
    HAS_GPU = True
except Exception as e:
    print("Exception loading Torch GPU library: ", e)
    # GPU import can fail, as Torch is not using cuda-lazy
    HAS_GPU = False
    pass


class THOpsTest(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           alpha=st.floats(min_value=0.1, max_value=2.0),
           in_place=st.booleans(),
           **(hu.gcs if HAS_GPU else hu.gcs_cpu_only))
    def test_elu(self, X, alpha, in_place, gc, dc):
        op = core.CreateOperator(
            "ELU",
            ["X"],
            ["X" if in_place else "Y"],
            engine="THNN",
            alpha=alpha)
        self.assertDeviceChecks(dc, op, [X], [0])

        def elu(X):
            Y = np.copy(X)
            Y[Y <= 0] = (np.exp(Y[Y <= 0]) - 1) * alpha
            return (Y,)

        self.assertReferenceChecks(gc, op, [X], elu)
        # Avoid the nonlinearity at 0 for gradient checker.
        X[X == 0] += 0.2
        X[np.abs(X) < 0.2] += np.sign(X[np.abs(X) < 0.2])
        assert len(X[np.abs(X) < 0.2]) == 0
        self.assertGradientChecks(gc, op, [X], 0, [0])
