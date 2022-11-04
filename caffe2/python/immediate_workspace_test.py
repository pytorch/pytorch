import os

import numpy as np
from caffe2.python import core, test_util, workspace

# This test is outside of workspace_test.py because it changes the root folder of the
# workspace, and if this is ordered above other tests, it will impact what the future
# root folder will be, even after resets.
class TestImmediate(test_util.TestCase):
    def testImmediateEnterExit(self):
        workspace.StartImmediate(i_know=True)
        self.assertTrue(workspace.IsImmediate())
        workspace.StopImmediate()
        self.assertFalse(workspace.IsImmediate())

    def testImmediateRunsCorrectly(self):
        workspace.StartImmediate(i_know=True)
        net = core.Net("test-net")
        net.ConstantFill([], "testblob", shape=[1, 2, 3, 4], value=1.0)
        self.assertEqual(workspace.ImmediateBlobs(), ["testblob"])
        content = workspace.FetchImmediate("testblob")
        # Also, the immediate mode should not invade the original namespace,
        # so we check if this is so.
        with self.assertRaises(RuntimeError):
            workspace.FetchBlob("testblob")
        np.testing.assert_array_equal(content, 1.0)
        content[:] = 2.0
        self.assertTrue(workspace.FeedImmediate("testblob", content))
        np.testing.assert_array_equal(workspace.FetchImmediate("testblob"), 2.0)
        workspace.StopImmediate()
        with self.assertRaises(RuntimeError):
            content = workspace.FetchImmediate("testblob")

    def testImmediateRootFolder(self):
        workspace.StartImmediate(i_know=True)
        # for testing we will look into the _immediate_root_folder variable
        # but in normal usage you should not access that.
        self.assertTrue(len(workspace._immediate_root_folder) > 0)
        root_folder = workspace._immediate_root_folder
        self.assertTrue(os.path.isdir(root_folder))
        workspace.StopImmediate()
        self.assertTrue(len(workspace._immediate_root_folder) == 0)
        # After termination, immediate mode should have the root folder
        # deleted.
        self.assertFalse(os.path.exists(root_folder))
