from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import numpy as np

class TestLog1pOp(hu.HypothesisTestCase):

    @given(data=hu.tensor(dtype=np.float32), **hu.gcs)
    def test_log1p(self, data, gc, dc):
        op = core.CreateOperator("Log1p", ["input"], ["output"])

        def ref_log1p(input):
            result = np.log1p(input)
            return (result,)

        self.assertReferenceChecks(gc, op, [data], ref_log1p)
