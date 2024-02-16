




import unittest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class DropoutTest(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           in_place=st.booleans(),
           ratio=st.floats(0, 0.999),
           **mu.gcs)
    def test_dropout_is_test(self, X, in_place, ratio, gc, dc):
        """Test with is_test=True for a deterministic reference impl."""
        op = core.CreateOperator('Dropout', ['X'],
                                 ['X' if in_place else 'Y'],
                                 ratio=ratio, is_test=True)

        self.assertDeviceChecks(dc, op, [X], [0])
        # No sense in checking gradients for test phase

        def reference_dropout_test(x):
            return x, np.ones(x.shape, dtype=bool)
        self.assertReferenceChecks(
            gc, op, [X], reference_dropout_test,
            # The 'mask' output may be uninitialized
            outputs_to_check=[0])

    @given(X=hu.tensor(),
           in_place=st.booleans(),
           output_mask=st.booleans(),
           **mu.gcs)
    @unittest.skipIf(True, "Skip duo to different rand seed.")
    def test_dropout_ratio0(self, X, in_place, output_mask, gc, dc):
        """Test with ratio=0 for a deterministic reference impl."""
        is_test = not output_mask
        op = core.CreateOperator('Dropout', ['X'],
                                 ['X' if in_place else 'Y'] +
                                 (['mask'] if output_mask else []),
                                 ratio=0.0, is_test=is_test)

        self.assertDeviceChecks(dc, op, [X], [0])

        def reference_dropout_ratio0(x):
            return (x,) if is_test else (x, np.ones(x.shape, dtype=bool))
        self.assertReferenceChecks(
            gc, op, [X], reference_dropout_ratio0, outputs_to_check=[0])

if __name__ == "__main__":
    unittest.main()
