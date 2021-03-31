from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import numpy as np

def ref_log1p(input):
    result = np.log1p(input)
    return (result,)

def ref_log1p_grad(g_out, outputs, fwd_inputs):
    result = g_out / (fwd_inputs[0] + 1)
    return (result,)

class TestLog1pOp(hu.HypothesisTestCase):

    @given(input=hu.tensor(dtype=np.float32), **hu.gcs)
    def test_log1p(self, input, gc, dc):
        op = core.CreateOperator("Log1p", ["input"], ["output"])

        self.assertReferenceChecks(gc, op, [input], ref_log1p,
                                   output_to_grad="output",
                                   grad_reference=ref_log1p_grad,
                                   ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [input], [0])
