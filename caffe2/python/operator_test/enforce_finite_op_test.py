




from hypothesis import given, settings
import numpy as np

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st


class TestEnforceFinite(hu.HypothesisTestCase):
    @given(
        X=hu.tensor(
            # allow empty
            min_value=0,
            elements=hu.floats(allow_nan=True, allow_infinity=True),
        ),
        **hu.gcs
    )
    @settings(deadline=10000)
    def test_enforce_finite(self, X, gc, dc):

        def all_finite_value(X):
            if X.size <= 0:
                return True

            return np.isfinite(X).all()

        net = core.Net('test_net')
        net.Const(array=X, blob_out="X")
        net.EnforceFinite("X", [])

        if all_finite_value(X):
            self.assertTrue(workspace.RunNetOnce(net))
        else:
            with self.assertRaises(RuntimeError):
                workspace.RunNetOnce(net)

    @given(
        X=hu.tensor(
            elements=hu.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
        ),
        **hu.gcs
    )
    def test_enforce_finite_device_check(self, X, gc, dc):
        op = core.CreateOperator(
            "EnforceFinite",
            ["X"],
            [],
        )
        self.assertDeviceChecks(dc, op, [X], [])
