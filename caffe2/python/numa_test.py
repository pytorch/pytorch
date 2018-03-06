from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.test_util import TestCase
import unittest

core.GlobalInit(["caffe2", "--caffe2_cpu_numa_enabled=1"])


@unittest.skipIf(not workspace.IsNUMAEnabled(), "NUMA is not enabled")
@unittest.skipIf(workspace.GetNumNUMANodes() < 2, "Not enough NUMA nodes")
class NUMATest(TestCase):
    def test_numa(self):
        net = core.Net("test_numa")
        net.Proto().type = "async_scheduling"

        numa_device_option = caffe2_pb2.DeviceOption()
        numa_device_option.device_type = caffe2_pb2.CPU
        numa_device_option.numa_node_id = 0

        net.ConstantFill([], "output_blob_0", shape=[1], value=3.14,
                             device_option=numa_device_option)

        numa_device_option.numa_node_id = 1
        net.ConstantFill([], "output_blob_1", shape=[1], value=3.14,
                             device_option=numa_device_option)

        workspace.RunNetOnce(net)

        self.assertEqual(workspace.GetBlobNUMANode("output_blob_0"), 0)
        self.assertEqual(workspace.GetBlobNUMANode("output_blob_1"), 1)


if __name__ == '__main__':
    unittest.main()
