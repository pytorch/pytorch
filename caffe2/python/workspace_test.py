import numpy as np
import os
import unittest

from caffe2.python import core, test_util, workspace


class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.net = core.Net("test-net")
        self.testblob_ref = self.net.ConstantFill(
            [], "testblob", shape=[1, 2, 3, 4], value=1.0)
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

    def testFetchFeedBlobViaBlobReference(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        fetched = workspace.FetchBlob(self.testblob_ref)
        # check if fetched is correct.
        self.assertEqual(fetched.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched, 1.0)
        fetched[:] = 2.0
        self.assertEqual(workspace.FeedBlob(self.testblob_ref, fetched), True)
        fetched_again = workspace.FetchBlob("testblob")  # fetch by name now
        self.assertEqual(fetched_again.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched_again, 2.0)


    def testFetchFeedBlobTypes(self):
        for dtype in [np.float16, np.float32, np.float64, np.bool,
                      np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16]:
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

    def testFetchFeedBlobBool(self):
        """Special case for bool to ensure coverage of both true and false."""
        data = np.zeros((2, 3, 4)).astype(np.bool)
        data.flat[::2] = True
        self.assertEqual(workspace.FeedBlob("testblob_types", data), True)
        fetched_back = workspace.FetchBlob("testblob_types")
        self.assertEqual(fetched_back.shape, (2, 3, 4))
        self.assertEqual(fetched_back.dtype, np.bool)
        np.testing.assert_array_equal(fetched_back, data)

    def testFetchFeedBlobZeroDim(self):
        data = np.empty(shape=(2, 0, 3), dtype=np.float32)
        self.assertEqual(workspace.FeedBlob("testblob_empty", data), True)
        fetched_back = workspace.FetchBlob("testblob_empty")
        self.assertEqual(fetched_back.shape, (2, 0, 3))
        self.assertEqual(fetched_back.dtype, np.float32)

    def testFetchFeedLongStringTensor(self):
        # long strings trigger array of object creation
        strs = np.array([
            ' '.join(10 * ['long string']),
            ' '.join(128 * ['very long string']),
            'small \0\1\2 string',
            "Hello, world! I have special \0 symbols \1!"])
        workspace.FeedBlob('my_str_tensor', strs)
        strs2 = workspace.FetchBlob('my_str_tensor')
        self.assertEqual(strs.shape, strs2.shape)
        for i in range(0, strs.shape[0]):
            self.assertEqual(strs[i], strs2[i])

    def testFetchFeedShortStringTensor(self):
        # small strings trigger NPY_STRING array
        strs = np.array(['elem1', 'elem 2', 'element 3'])
        workspace.FeedBlob('my_str_tensor_2', strs)
        strs2 = workspace.FetchBlob('my_str_tensor_2')
        self.assertEqual(strs.shape, strs2.shape)
        for i in range(0, strs.shape[0]):
            self.assertEqual(strs[i], strs2[i])

    def testFetchFeedPlainString(self):
        # this is actual string, not a tensor of strings
        s = "Hello, world! I have special \0 symbols \1!"
        workspace.FeedBlob('my_plain_string', s)
        s2 = workspace.FetchBlob('my_plain_string')
        self.assertEqual(s, s2)

    def testFetchFeedViaBlobDict(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        fetched = workspace.blobs["testblob"]
        # check if fetched is correct.
        self.assertEqual(fetched.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched, 1.0)
        fetched[:] = 2.0
        workspace.blobs["testblob"] = fetched
        fetched_again = workspace.blobs["testblob"]
        self.assertEqual(fetched_again.shape, (1, 2, 3, 4))
        np.testing.assert_array_equal(fetched_again, 2.0)

        self.assertTrue("testblob" in workspace.blobs)
        self.assertFalse("non_existant" in workspace.blobs)
        self.assertEqual(len(workspace.blobs), 1)
        for key in workspace.blobs:
            self.assertEqual(key, "testblob")


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


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
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
        self.assertEqual(pattern.shape[0], workspace.NumCudaDevices())


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
        self.assertEqual(
            workspace.ImmediateBlobs(), ["testblob"])
        content = workspace.FetchImmediate("testblob")
        # Also, the immediate mode should not invade the original namespace,
        # so we check if this is so.
        with self.assertRaises(KeyError):
            workspace.FetchBlob("testblob")
        np.testing.assert_array_equal(content, 1.0)
        content[:] = 2.0
        self.assertTrue(workspace.FeedImmediate("testblob", content))
        np.testing.assert_array_equal(
            workspace.FetchImmediate("testblob"), 2.0)
        workspace.StopImmediate()
        with self.assertRaises(KeyError):
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


class TestCppEnforceAsException(test_util.TestCase):
    def testEnforce(self):
        op = core.CreateOperator("Relu", ["X"], ["Y"])
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

if __name__ == '__main__':
    unittest.main()
