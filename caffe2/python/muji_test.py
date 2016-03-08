import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from pycaffe2 import core, workspace, muji

class TestMuji(unittest.TestCase):
  """Test class for Muji."""

  def RunningAllreduceWithGPUs(self, gpu_ids, allreduce_function):
    """A base function to test different scenarios."""
    workspace.ResetWorkspace()
    net = core.Net("mujitest")
    for id in gpu_ids:
      net.ConstantFill([], "testblob_gpu_" + str(id), shape=[1, 2, 3, 4],
                            value=float(id+1),
                            device_option=muji.OnGPU(id))
    allreduce_function(
      net, ["testblob_gpu_" + str(i) for i in gpu_ids],
      "_reduced", gpu_ids)
    workspace.RunNetOnce(net)
    target_value = sum(gpu_ids) + len(gpu_ids)
    all_blobs = workspace.Blobs()
    all_blobs.sort()
    for blob in all_blobs:
      print blob, workspace.FetchBlob(blob)

    for id in gpu_ids:
      blob = workspace.FetchBlob("testblob_gpu_" + str(i) + "_reduced")
      np.testing.assert_array_equal(
          blob, target_value, err_msg="gpu id %d of %s" % (id, str(gpu_ids)))


  def testAllreduceFallback(self):
    self.RunningAllreduceWithGPUs(
        range(workspace.NumberOfGPUs()), muji.AllreduceFallback)

  def testAllreduceSingleGPU(self):
    for i in range(workspace.NumberOfGPUs()):
      self.RunningAllreduceWithGPUs([i], muji.Allreduce)

  def testAllreduceWithTwoGPUs(self):
    pattern = workspace.GetCudaPeerAccessPattern()
    if pattern.shape[0] >= 2 and np.all(pattern[:2,:2]):
      self.RunningAllreduceWithGPUs([0, 1], muji.Allreduce2)
    else:
      print 'Skipping allreduce with 2 gpus. Not peer access ready.'

  def testAllreduceWithFourGPUs(self):
    pattern = workspace.GetCudaPeerAccessPattern()
    if pattern.shape[0] >= 4 and np.all(pattern[:4,:4]):
      self.RunningAllreduceWithGPUs([0, 1, 2, 3], muji.Allreduce4)
    else:
      print 'Skipping allreduce with 4 gpus. Not peer access ready.'

  def testAllreduceWithEightGPUs(self):
    pattern = workspace.GetCudaPeerAccessPattern()
    if (pattern.shape[0] >= 8 and np.all(pattern[:4,:4])
        and np.all(pattern[4:, 4:])):
      self.RunningAllreduceWithGPUs(range(8), muji.Allreduce8)
    else:
      print 'Skipping allreduce with 8 gpus. Not peer access ready.'

if __name__ == '__main__':
  if not workspace.has_gpu_support:
    print 'No GPU support. skipping muji test.'
  elif workspace.NumberOfGPUs() == 0:
    print 'No GPU device. Skipping gpu test.'
  else:
    workspace.GlobalInit(['python'])
    unittest.main()
