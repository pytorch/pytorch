




from caffe2.python import workspace, core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np


class TestNegateGradient(serial.SerializedTestCase):

    @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_forward(self, X, inplace, gc, dc):
        def neg_grad_ref(X):
            return (X,)

        op = core.CreateOperator("NegateGradient", ["X"], ["Y" if not inplace else "X"])
        self.assertReferenceChecks(gc, op, [X], neg_grad_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(size=st.lists(st.integers(min_value=1, max_value=20),
                         min_size=1, max_size=5))
    def test_grad(self, size):
        X = np.random.random_sample(size)
        workspace.ResetWorkspace()
        workspace.FeedBlob("X", X.astype(np.float32))

        net = core.Net("negate_grad_test")
        Y = net.NegateGradient(["X"], ["Y"])

        grad_map = net.AddGradientOperators([Y])
        workspace.RunNetOnce(net)

        # check X_grad == negate of Y_grad
        x_val, y_val = workspace.FetchBlobs(['X', 'Y'])
        x_grad_val, y_grad_val = workspace.FetchBlobs([grad_map['X'],
                                                        grad_map['Y']])
        np.testing.assert_array_equal(x_val, y_val)
        np.testing.assert_array_equal(x_grad_val, y_grad_val * (-1))
