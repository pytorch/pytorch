import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.net = core.Net("test-net")
        self.net.ConstantFill([], "testblob", shape=[1, 2, 3, 4], value=1.0)
        workspace.ResetWorkspace()

    def testRootFolder(self):
        self.assertEqual(workspace.ResetWorkspace(), True)
        self.assertEqual(workspace.RootFolder(), ".")
        self.assertEqual(
            workspace.ResetWorkspace("/tmp/caffe-workspace-test"), True)
        self.assertEqual(workspace.RootFolder(), "/tmp/caffe-workspace-test")

    def testWorkspaceHasBlobWithNonexistingName(self):
        self.assertEqual(workspace.HasBlob("non-existing"), False)

    def testRunOperatorOnce(self):
        self.assertEqual(
            workspace.RunOperatorOnce(
                self.net.Proto().op[0].SerializeToString()
            ), True
        )
        self.assertEqual(workspace.HasBlob("testblob"), True)
        blobs = workspace.Blobs()
        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0], "testblob")

    def testRunNetOnce(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)

    def testRunPlan(self):
        plan = core.Plan("test-plan")
        plan.AddNets([self.net])
        plan.AddStep(core.ExecutionStep("test-step", self.net))
        self.assertEqual(
            workspace.RunPlan(plan.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)

    def testResetWorkspace(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)
        self.assertEqual(workspace.ResetWorkspace(), True)
        self.assertEqual(workspace.HasBlob("testblob"), False)

    def testFetchFeedBlob(self):
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

    def testFetchFeedBlobTypes(self):
        for dtype in [np.float32, np.int32, np.int64, np.float16]:
            try:
                rng = np.iinfo(dtype).max * 2
            except ValueError:
                rng = 1000
            data = ((np.random.rand(2, 3, 4) - 0.5) * rng).astype(dtype)
            self.assertEqual(workspace.FeedBlob("testblob_types", data), True)
            fetched_back = workspace.FetchBlob("testblob_types")
            self.assertEqual(fetched_back.shape, (2, 3, 4))
            self.assertEqual(fetched_back.dtype, dtype)
            np.testing.assert_array_equal(fetched_back, data)


class TestMultiWorkspaces(unittest.TestCase):
    def setUp(self):
        workspace.SwitchWorkspace("default")
        workspace.ResetWorkspace()

    def testCreateWorkspace(self):
        self.net = core.Net("test-net")
        self.net.ConstantFill([], "testblob", shape=[1, 2, 3, 4], value=1.0)
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True
        )
        self.assertEqual(workspace.HasBlob("testblob"), True)
        self.assertEqual(workspace.SwitchWorkspace("test", True), True)
        self.assertEqual(workspace.HasBlob("testblob"), False)
        self.assertEqual(workspace.SwitchWorkspace("default"), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)

        try:
            # The following should raise an error.
            workspace.SwitchWorkspace("non-existing")
            # so this should never happen.
            self.assertEqual(True, False)
        except RuntimeError:
            pass

        workspaces = workspace.Workspaces()
        self.assertTrue("default" in workspaces)
        self.assertTrue("test" in workspaces)


if __name__ == '__main__':
    unittest.main()
