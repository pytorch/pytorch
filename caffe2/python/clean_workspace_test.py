import unittest

from caffe2.python import workspace


# This test is extracted out from workspace_test.py because it relies on the pristine
# state of the initial workspace. When tests are run in different orders, this test may
# become flaky because of global state modifications impacting what the root folder is
# after a reset.
class TestWorkspace(unittest.TestCase):
    def testRootFolder(self):
        self.assertEqual(workspace.ResetWorkspace(), True)
        self.assertEqual(workspace.RootFolder(), ".")
        self.assertEqual(workspace.ResetWorkspace("/tmp/caffe-workspace-test"), True)
        self.assertEqual(workspace.RootFolder(), "/tmp/caffe-workspace-test")
