from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python.core import CreatePythonOperator
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest

class PythonOpTest(hu.HypothesisTestCase):
    @given(x=hu.tensor(),
           n=st.integers(min_value=1, max_value=20),
           w=st.integers(min_value=1, max_value=20))
    def test_simple_python_op(self, x, n, w):
        def g(input_, output):
            output[...] = input_

        def f(inputs, outputs):
            outputs[0].reshape(inputs[0].shape)
            g(inputs[0].data, outputs[0].data)

        ops = [CreatePythonOperator(f, ["x"], [str(i)]) for i in range(n)]
        net = core.Net("net")
        net.Proto().op.extend(ops)
        net.Proto().type = "dag"
        net.Proto().num_workers = w
        iters = 100
        plan = core.Plan("plan")
        plan.AddStep(core.ExecutionStep("test-step", net, iters))
        workspace.FeedBlob("x", x)
        workspace.RunPlan(plan.Proto().SerializeToString())
        for i in range(n):
            y = workspace.FetchBlob(str(i))
            np.testing.assert_almost_equal(x, y)


if __name__ == "__main__":
    import unittest
    unittest.main()
