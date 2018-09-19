from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestPad(serial.SerializedTestCase):
    @serial.given(pad_t=st.integers(-5, 0),
           pad_l=st.integers(-5, 0),
           pad_b=st.integers(-5, 0),
           pad_r=st.integers(-5, 0),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           size_w=st.integers(16, 128),
           size_h=st.integers(16, 128),
           size_c=st.integers(1, 4),
           size_n=st.integers(1, 4),
           **hu.gcs)
    def test_crop(self,
                  pad_t, pad_l, pad_b, pad_r,
                  mode,
                  size_w, size_h, size_c, size_n,
                  gc, dc):
        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
        )
        X = np.random.rand(
            size_n, size_c, size_h, size_w).astype(np.float32)

        def ref(X):
            return (X[:, :, -pad_t:pad_b or None, -pad_l:pad_r or None],)

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
