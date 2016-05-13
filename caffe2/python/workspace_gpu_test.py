import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, test_util


@unittest.skipIf(
    not workspace.has_gpu_support or workspace.NumberOfGPUs() == 0,
    "No gpu support.")
class TestWorkspaceGPU(test_util.TestCase):

    def setUp(self):
        workspace.ResetWorkspace()
        self.net = core.Net("test-net")
        self.net.ConstantFill([], "testblob", shape=[1, 2, 3, 4], value=1.0)
        self.net.RunAllOnGPU()

    def testFetchBlobGPU(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        fetched = workspace.FetchBlob("testblob")
        # check if fetched is correct.
        self.assertEqual(fetched.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched, 1.0)
        fetched[:] = 2.0
        self.assertEqual(workspace.FeedBlob("testblob", fetched), True)
        fetched_again = workspace.FetchBlob("testblob")
        self.assertEqual(fetched_again.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched_again, 2.0)

    def testDefaultGPUID(self):
        self.assertEqual(workspace.SetDefaultGPUID(0), True)
        self.assertEqual(workspace.GetDefaultGPUID(), 0)

    def testGetCudaPeerAccessPattern(self):
        pattern = workspace.GetCudaPeerAccessPattern()
        self.assertEqual(type(pattern), np.ndarray)
        self.assertEqual(pattern.ndim, 2)
        self.assertEqual(pattern.shape[0], pattern.shape[1])
        self.assertEqual(pattern.shape[0], workspace.NumberOfGPUs())


if __name__ == '__main__':
    unittest.main()
