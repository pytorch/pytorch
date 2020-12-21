




from hypothesis import given
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestFlatten(hu.HypothesisTestCase):
    @given(X=hu.tensor(min_dim=2, max_dim=4),
           **hu.gcs)
    def test_flatten(self, X, gc, dc):
        for axis in range(X.ndim + 1):
            op = core.CreateOperator(
                "Flatten",
                ["X"],
                ["Y"],
                axis=axis)

            def flatten_ref(X):
                shape = X.shape
                outer = np.prod(shape[:axis]).astype(int)
                inner = np.prod(shape[axis:]).astype(int)
                return np.copy(X).reshape(outer, inner),

            self.assertReferenceChecks(gc, op, [X], flatten_ref)

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
