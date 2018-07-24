from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given


class TestPad(hu.HypothesisTestCase):
    @given(pads_1_begin=st.integers(0, 1),
           pads_1_end=st.integers(0, 1),
           pads_2_begin=st.integers(0, 2),
           pads_2_end=st.integers(0, 2),
           pads_3_begin=st.integers(0, 2),
           pads_3_end=st.integers(0, 2),
           mode=st.sampled_from(["constant"]),
           value=st.floats(0.0, 0.0),
           size_n1=st.integers(4, 4),
           size_n2=st.integers(5, 5),
           size_n3=st.integers(6, 6),
           **hu.gcs)
    def test_crop(self,
                  pads_1_begin, pads_2_begin, pads_3_begin,
                  pads_1_end, pads_2_end, pads_3_end,
                  mode, value,
                  size_n1, size_n2, size_n3,
                  gc, dc):
        pads = [pads_1_begin, pads_2_begin, pads_3_begin,
                pads_1_end, pads_2_end, pads_3_end]

        op = core.CreateOperator(
            "Pad",
            ["X"],
            ["Y"],
            value=value,
            mode=mode,
            pads=pads,
        )
        X = np.random.rand(
            size_n1, size_n2, size_n3).astype(np.float32)

        def ref(X):
            return (np.pad(X, ((pads_1_begin, pads_1_end),
                               (pads_2_begin, pads_2_end),
                               (pads_3_begin, pads_3_end)), mode),)


        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        # self.assertGradientChecks(gc, op, [X], 0, [0])

if __name__ == "__main__":
    unittest.main()
