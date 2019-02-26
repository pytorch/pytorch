from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import unittest

device_opts = caffe2_pb2.DeviceOption()

class TestGatherOpCPU(serial.SerializedTestCase):
    device_type = caffe2_pb2.CPU

    def setUp(self):
           device_opts.device_type = self.device_type

    def run_gather(self, data, inds, dtype):
        op = core.CreateOperator(
            "Gather",
            ["DATA", "INDICES"],
            ["OUTPUT"],
            "test gather operation",
            control_input = None,
            device_option=device_opts
            )
        workspace.ResetWorkspace()
        workspace.FeedBlob("DATA", data.astype(dtype),device_opts)
        workspace.FeedBlob("INDICES", inds.astype(np.int32),device_opts)
        workspace.RunOperatorOnce(op)
        expected = np.take(data,inds);
        np.testing.assert_array_equal(expected,workspace.FetchBlob("OUTPUT"))

    def test_gather_int8(self):
        data = np.array([1, 2, 3, 4],dtype=np.int8)
        inds = np.array([3,2,1,0])
        self.run_gather(data,inds,np.int8)

    def test_gather_float64(self):
        data = np.array([1.3482698511467371e+308, -1.3482698511467371e+308,
                         0.9999999999999999, 5e-324],dtype=np.float64)
        inds = np.array([3,0,1,2])
        self.run_gather(data,inds,np.float64)

    def test_gather_double(self):
        data = np.array([1.3482698511467371e+308, -1.3482698511467371e+308,
                         0.9999999999999999, 5e-324],dtype=np.double)
        inds = np.array([3,0,1,2])
        self.run_gather(data,inds,np.double)

    def test_gather_fp16(self):
        data = np.array([1.01,0.99951,65504,0.000061035],dtype=np.float16)
        inds = np.array([3,0,1,2])
        self.run_gather(data,inds,np.float16)

    def test_gather_long(self):
        data = np.array([1.01,0.99951,65504,0.000061035],dtype=np.long)
        inds = np.array([3,0,1,2])
        self.run_gather(data,inds,np.long)

    def test_gather_float(self):
        data = np.array([1.4012984645e-45,3.4028234664e+38,
                         0.9999999404,1.0000001192],dtype=np.float)
        inds = np.array([3,0,1,2])
        self.run_gather(data,inds,np.float)

# run same tests on GPU if available
if(workspace.has_gpu_support):
    TestGatherOpGPU = type(str("TestTestGatherOpGPU"),
                              (unittest.TestCase,),
                              dict(TestGatherOpCPU.__dict__,
                                  device_type = caffe2_pb2.CUDA))


if __name__ == "__main__":
    unittest.main()
