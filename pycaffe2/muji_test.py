import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from pycaffe2 import core, workspace, muji

class TestMuji(unittest.TestCase):
  def setUp(self):
    self.net = core.Net("test-net")
    for i in range(workspace.NumberOfGPUs()):
      self.net.ConstantFill([], "testblob_gpu_" + str(i), shape=[1, 2, 3, 4],
                            value=float(i+1),
                            device_option=muji.CudaDeviceOption(i))

  def testRunningRawNet(self):
    workspace.RunNetOnce(self.net)
    for i in range(workspace.NumberOfGPUs()):
      blob = workspace.FetchBlob("testblob_gpu_" + str(i))
      np.testing.assert_array_equal(blob, i + 1)

  def testRunningAllreduce(self):
    total_gpus = workspace.NumberOfGPUs()
    target_value = (total_gpus * (total_gpus + 1)) / 2
    # Add operators for multigpu
    muji.Allreduce(
        self.net,
        ["testblob_gpu_" + str(i) for i in range(total_gpus)],
        "_reduced")
    workspace.RunNetOnce(self.net)
    for i in range(total_gpus):
      blob = workspace.FetchBlob("testblob_gpu_" + str(i) + "_reduced")
      np.testing.assert_array_equal(blob, target_value)


if __name__ == '__main__':
  if not workspace.has_gpu_support:
    print 'No GPU support. skipping muji test.'
  else:
    unittest.main()
