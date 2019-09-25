from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

import unittest


class TestCeil(serial.SerializedTestCase):

    @serial.given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_ceil(self, X, gc, dc, engine):
        op = core.CreateOperator("Ceil", ["X"], ["Y"], engine=engine)

        def ceil_ref(X):
            return (np.ceil(X),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ceil_ref)

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
