




import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, dyndep, workspace

dyndep.InitOpsLibrary("@/caffe2/caffe2/contrib/prof:cuda_profile_ops")


class CudaProfileOpsTest(unittest.TestCase):
    @unittest.skipIf(workspace.NumCudaDevices() < 1, "Need at least 1 GPU")
    def test_run(self):
        net = core.Net("net")
        net.CudaProfileInitialize([], [], output="/tmp/cuda_profile_test")
        net.CudaProfileStart([], [])
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            net.ConstantFill([], ["out"], shape=[1, 3, 244, 244])
        net.CudaProfileStop([], [])

        workspace.CreateNet(net)
        workspace.RunNet(net)
