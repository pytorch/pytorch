from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.test_util import TestCase
import unittest

core.GlobalInit(["caffe2", "--caffe2_cpu_numa_enabled=1"])


def build_test_net(net_name):
    net = core.Net(net_name)
    net.Proto().type = "async_scheduling"

    numa_device_option = caffe2_pb2.DeviceOption()
    numa_device_option.device_type = caffe2_pb2.CPU
    numa_device_option.numa_node_id = 0

    net.ConstantFill([], "output_blob_0", shape=[1], value=3.14,
                         device_option=numa_device_option)

    numa_device_option.numa_node_id = 1
    net.ConstantFill([], "output_blob_1", shape=[1], value=3.14,
                         device_option=numa_device_option)

    gpu_device_option = caffe2_pb2.DeviceOption()
    gpu_device_option.device_type = caffe2_pb2.CUDA
    gpu_device_option.cuda_gpu_id = 0

    net.CopyCPUToGPU("output_blob_0", "output_blob_0_gpu",
                        device_option=gpu_device_option)
    net.CopyCPUToGPU("output_blob_1", "output_blob_1_gpu",
                        device_option=gpu_device_option)

    return net


@unittest.skipIf(not workspace.IsNUMAEnabled(), "NUMA is not enabled")
@unittest.skipIf(workspace.GetNumNUMANodes() < 2, "Not enough NUMA nodes")
@unittest.skipIf(not workspace.has_gpu_support, "No GPU support")
class NUMATest(TestCase):
    def test_numa(self):
        net = build_test_net("test_numa")

        workspace.RunNetOnce(net)

        self.assertEqual(workspace.GetBlobNUMANode("output_blob_0"), 0)
        self.assertEqual(workspace.GetBlobNUMANode("output_blob_1"), 1)


if __name__ == '__main__':
    unittest.main()
