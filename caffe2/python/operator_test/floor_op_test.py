




import unittest

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from caffe2.python import core


class TestFloor(serial.SerializedTestCase):

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_floor(self, X, gc, dc, engine):
        op = core.CreateOperator("Floor", ["X"], ["Y"], engine=engine)

        def floor_ref(X):
            return (np.floor(X),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=floor_ref)

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
