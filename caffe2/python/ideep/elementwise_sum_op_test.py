from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ElementwiseSumTest(hu.HypothesisTestCase):
    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inputs=st.integers(2, 7),
           inplace=st.booleans(),
           **mu.gcs)
    def test_elementwise_sum(self,
                                 size,
                                 input_channels,
                                 batch_size,
                                 inputs,
                                 inplace,
                                 gc,
                                 dc):
        op = core.CreateOperator(
            "Sum",
            ["X_{}".format(i) for i in range(inputs)],
            ["X_0" if inplace else "Y"],
        )
        Xs = [np.random.rand(batch_size, input_channels, size, size).astype(
            np.float32) for _ in range(inputs)]
        self.assertDeviceChecks(dc, op, Xs, [0])


    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inputs=st.integers(2, 7),
           inplace=st.booleans(),
           **mu.gcs)
    def test_elementwise_sum_fallback(self,
                                      size,
                                      input_channels,
                                      batch_size,
                                      inputs,
                                      inplace,
                                      gc,
                                      dc):
        op = core.CreateOperator(
            "Sum",
            ["X_{}".format(i) for i in range(inputs)],
            ["X_0" if inplace else "Y"],
            device_option=dc[1]
        )
        Xs = [np.random.rand(batch_size, input_channels, size, size).astype(
            np.float32) for _ in range(inputs)]

        sum_val = Xs[0]
        workspace.FeedBlob("X_0", Xs[0], dc[0])
        for i, x in enumerate(Xs):
            if i == 0: continue
            sum_val += x
            workspace.FeedBlob("X_{}".format(i), x, dc[1])

        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("X_0" if inplace else "Y")

        if not np.allclose(sum_val, Y, atol=0.01, rtol=0.01):
            print(Y.flatten())
            print(sum_val.flatten())
            print(np.max(np.abs(Y - sum_val)))
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
