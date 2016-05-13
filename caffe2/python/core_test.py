import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util


class TestScopes(test_util.TestCase):
    def testBlobReferenceIsIndependentFromNameScope(self):
        blob_v = core.BlobReference("v")
        with core.NameScope("foo"):
            blob_w = core.BlobReference("w")
            with core.NameScope("bar"):
                blob_x = core.BlobReference("x")
        self.assertEqual(str(blob_v), "v")
        self.assertEqual(str(blob_w), "w")
        self.assertEqual(str(blob_x), "x")

    def testNameScopeWithOp(self):
        global_x = core.BlobReference("x")
        global_y = core.BlobReference("y")
        with core.NameScope("foo"):
            # Raw strings should have namescope prepended.
            op = core.CreateOperator("Relu", "x", "y")
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "foo/x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "foo/y")
            # BlobReferences should not.
            op = core.CreateOperator("Relu", global_x, global_y)
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "y")

    def testDeviceScope(self):
        # No device
        op = core.CreateOperator("Relu", "x", "y")
        self.assertFalse(op.HasField('device_option'))
        # explicitly setting a device
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = 1
        op = core.CreateOperator("Relu", "x", "y", device_option=device_option)
        self.assertTrue(op.HasField('device_option'))
        self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
        self.assertEqual(op.device_option.cuda_gpu_id, 1)
        with core.DeviceScope(device_option):
            # from device scope
            op = core.CreateOperator("Relu", "x", "y")
            self.assertTrue(op.HasField('device_option'))
            self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
            self.assertEqual(op.device_option.cuda_gpu_id, 1)
            # from an overridden device option
            override_device = caffe2_pb2.DeviceOption()
            override_device.device_type = caffe2_pb2.CPU
            op = core.CreateOperator(
                "Relu", "x", "y", device_option=override_device)
            self.assertTrue(op.HasField('device_option'))
            self.assertEqual(op.device_option.device_type, caffe2_pb2.CPU)
        # back from normal: no device
        op = core.CreateOperator("Relu", "x", "y")
        self.assertFalse(op.HasField('device_option'))
        device_option = caffe2_pb2.DeviceOption()

    def testNameAndDeviceScopeTogether(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = 1
        with core.DeviceScope(device_option):
            with core.NameScope("foo"):
                op = core.CreateOperator("Relu", "x", "y")
                self.assertTrue(op.HasField('device_option'))
                self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
                self.assertEqual(op.device_option.cuda_gpu_id, 1)
                self.assertEqual(len(op.input), 1)
                self.assertEqual(op.input[0], "foo/x")
                self.assertEqual(len(op.output), 1)
                self.assertEqual(op.output[0], "foo/y")


if __name__ == '__main__':
    unittest.main()
