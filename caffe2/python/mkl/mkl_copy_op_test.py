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
import caffe2.python.mkl_test_util as mu
import caffe2.proto.caffe2_pb2 as pb2

@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKCopyTest(hu.HypothesisTestCase):
    @given(width=st.integers(7, 9),
           height=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           **mu.gcs)
    def test_mkl_copy(self,
                      width,
                      height,
                      input_channels,
                      batch_size,
                      gc, dc):
        X = np.random.rand(
            batch_size, input_channels, width, height).astype(np.float32)
        self.ws.create_blob("X").feed(X, pb2.DeviceOption())
        self.ws.run(core.CreateOperator(
            "CopyCPUToMKL",
            ["X"],
            ["X_MKL"],
            device_option=pb2.DeviceOption(device_type=pb2.MKLDNN)
        ))
        self.ws.run(core.CreateOperator(
            "CopyMKLToCPU",
            ["X_MKL"],
            ["X_copy"],
            device_option=pb2.DeviceOption(device_type=pb2.MKLDNN)
        ))
        np.testing.assert_array_equal(X, self.ws.blobs["X_copy"].fetch())

    @given(n=st.sampled_from([0, 10]))
    def test_mkl_zero_copy(self, n):
        shape = (0, n)
        X = np.zeros(shape=shape).astype(np.float32)
        self.ws.create_blob("X").feed(X, pb2.DeviceOption())
        self.ws.run(core.CreateOperator(
            "CopyCPUToMKL",
            ["X"],
            ["X_MKL"],
            device_option=pb2.DeviceOption(device_type=pb2.MKLDNN)
        ))
        self.ws.run(core.CreateOperator(
            "CopyMKLToCPU",
            ["X_MKL"],
            ["X_copy"],
            device_option=pb2.DeviceOption(device_type=pb2.MKLDNN)
        ))
        np.testing.assert_equal(shape, self.ws.blobs["X_copy"].fetch().shape)


if __name__ == "__main__":
    import unittest
    unittest.main()
