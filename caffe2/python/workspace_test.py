from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core, test_util, workspace, model_helper, brew

import caffe2.python.hypothesis_test_util as htu
import hypothesis.strategies as st
from hypothesis import given


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

    def testCurrentWorkspaceWrapper(self):
        self.assertNotIn("testblob", workspace.C.Workspace.current.blobs)
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)
        self.assertIn("testblob", workspace.C.Workspace.current.blobs)
        workspace.ResetWorkspace()
        self.assertNotIn("testblob", workspace.C.Workspace.current.blobs)

    def testRunPlan(self):
        plan = core.Plan("test-plan")
        plan.AddStep(core.ExecutionStep("test-step", self.net))
        self.assertEqual(
            workspace.RunPlan(plan.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)

    def testConstructPlanFromSteps(self):
        step = core.ExecutionStep("test-step-as-plan", self.net)
        self.assertEqual(workspace.RunPlan(step), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)

    def testResetWorkspace(self):
        self.assertEqual(
            workspace.RunNetOnce(self.net.Proto().SerializeToString()), True)
        self.assertEqual(workspace.HasBlob("testblob"), True)
        self.assertEqual(workspace.ResetWorkspace(), True)
        self.assertEqual(workspace.HasBlob("testblob"), False)

    def testTensorAccess(self):
        ws = workspace.C.Workspace()

        """ test in-place modification """
        ws.create_blob("tensor").feed(np.array([1.1, 1.2, 1.3]))
        tensor = ws.blobs["tensor"].tensor()
        tensor.data[0] = 3.3
        val = np.array([3.3, 1.2, 1.3])
        np.testing.assert_array_equal(tensor.data, val)
        np.testing.assert_array_equal(ws.blobs["tensor"].fetch(), val)

        """ test in-place initialization """
        tensor.init([2, 3], core.DataType.INT32)
        tensor.data[1, 1] = 100
        val = np.zeros([2, 3], dtype=np.int32)
        val[1, 1] = 100
        np.testing.assert_array_equal(tensor.data, val)
        np.testing.assert_array_equal(ws.blobs["tensor"].fetch(), val)

        """ strings cannot be initialized from python """
        with self.assertRaises(RuntimeError):
            tensor.init([3, 4], core.DataType.STRING)

        """ feed (copy) data into tensor """
        val = np.array([[b'abc', b'def'], [b'ghi', b'jkl']], dtype=np.object)
        tensor.feed(val)
        self.assertEquals(tensor.data[0, 0], b'abc')
        np.testing.assert_array_equal(ws.blobs["tensor"].fetch(), val)

        val = np.array([1.1, 10.2])
        tensor.feed(val)
        val[0] = 5.2
        self.assertEquals(tensor.data[0], 1.1)

        """ fetch (copy) data from tensor """
        val = np.array([1.1, 1.2])
        tensor.feed(val)
        val2 = tensor.fetch()
        tensor.data[0] = 5.2
        val3 = tensor.fetch()
        np.testing.assert_array_equal(val, val2)
        self.assertEquals(val3[0], 5.2)

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
            b' '.join(10 * [b'long string']),
            b' '.join(128 * [b'very long string']),
            b'small \0\1\2 string',
            b"Hello, world! I have special \0 symbols \1!"])
        workspace.FeedBlob('my_str_tensor', strs)
        strs2 = workspace.FetchBlob('my_str_tensor')
        self.assertEqual(strs.shape, strs2.shape)
        for i in range(0, strs.shape[0]):
            self.assertEqual(strs[i], strs2[i])

    def testFetchFeedShortStringTensor(self):
        # small strings trigger NPY_STRING array
        strs = np.array([b'elem1', b'elem 2', b'element 3'])
        workspace.FeedBlob('my_str_tensor_2', strs)
        strs2 = workspace.FetchBlob('my_str_tensor_2')
        self.assertEqual(strs.shape, strs2.shape)
        for i in range(0, strs.shape[0]):
            self.assertEqual(strs[i], strs2[i])

    def testFetchFeedPlainString(self):
        # this is actual string, not a tensor of strings
        s = b"Hello, world! I have special \0 symbols \1!"
        workspace.FeedBlob('my_plain_string', s)
        s2 = workspace.FetchBlob('my_plain_string')
        self.assertEqual(s, s2)

    def testFetchBlobs(self):
        s1 = b"test1"
        s2 = b"test2"
        workspace.FeedBlob('s1', s1)
        workspace.FeedBlob('s2', s2)
        fetch1, fetch2 = workspace.FetchBlobs(['s1', 's2'])
        self.assertEquals(s1, fetch1)
        self.assertEquals(s2, fetch2)

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
        self.assertEqual(workspace.SwitchWorkspace("test", True), None)
        self.assertEqual(workspace.HasBlob("testblob"), False)
        self.assertEqual(workspace.SwitchWorkspace("default"), None)
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
        self.assertEqual(workspace.SetDefaultGPUID(0), None)
        self.assertEqual(workspace.GetDefaultGPUID(), 0)

    def testGetCudaPeerAccessPattern(self):
        pattern = workspace.GetCudaPeerAccessPattern()
        self.assertEqual(type(pattern), np.ndarray)
        self.assertEqual(pattern.ndim, 2)
        self.assertEqual(pattern.shape[0], pattern.shape[1])
        self.assertEqual(pattern.shape[0], workspace.NumCudaDevices())


@unittest.skipIf(not workspace.C.has_mkldnn, "No MKLDNN support.")
class TestWorkspaceMKLDNN(test_util.TestCase):

    def testFeedFetchBlobMKLDNN(self):
        arr = np.random.randn(2, 3).astype(np.float32)
        workspace.FeedBlob(
            "testblob_mkldnn", arr, core.DeviceOption(caffe2_pb2.MKLDNN))
        fetched = workspace.FetchBlob("testblob_mkldnn")
        np.testing.assert_array_equal(arr, fetched)


class TestImmedibate(test_util.TestCase):
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
        with self.assertRaises(RuntimeError):
            workspace.FetchBlob("testblob")
        np.testing.assert_array_equal(content, 1.0)
        content[:] = 2.0
        self.assertTrue(workspace.FeedImmediate("testblob", content))
        np.testing.assert_array_equal(
            workspace.FetchImmediate("testblob"), 2.0)
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


class TestCppEnforceAsException(test_util.TestCase):
    def testEnforce(self):
        op = core.CreateOperator("Relu", ["X"], ["Y"])
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)


class TestCWorkspace(htu.HypothesisTestCase):
    def test_net_execution(self):
        ws = workspace.C.Workspace()
        self.assertEqual(ws.nets, {})
        self.assertEqual(ws.blobs, {})
        net = core.Net("test-net")
        net.ConstantFill([], "testblob", shape=[1, 2, 3, 4], value=1.0)
        ws.create_net(net)
        # If we do not specify overwrite, this should raise an error.
        with self.assertRaises(RuntimeError):
            ws.create_net(net)
        # But, if we specify overwrite, this should pass.
        ws.create_net(net, True)
        # Overwrite can also be a kwarg.
        ws.create_net(net, overwrite=True)
        self.assertIn("testblob", ws.blobs)
        self.assertEqual(len(ws.nets), 1)
        net_name = net.Proto().name
        self.assertIn("test-net", net_name)
        net = ws.nets[net_name].run()
        blob = ws.blobs["testblob"]
        np.testing.assert_array_equal(
            np.ones((1, 2, 3, 4), dtype=np.float32),
            blob.fetch())

    @given(name=st.text(), value=st.floats(min_value=-1, max_value=1.0))
    def test_operator_run(self, name, value):
        ws = workspace.C.Workspace()
        op = core.CreateOperator(
            "ConstantFill", [], [name], shape=[1], value=value)
        ws.run(op)
        self.assertIn(name, ws.blobs)
        np.testing.assert_allclose(
            [value], ws.blobs[name].fetch(), atol=1e-4, rtol=1e-4)

    @given(blob_name=st.text(),
           net_name=st.text(),
           value=st.floats(min_value=-1, max_value=1.0))
    def test_net_run(self, blob_name, net_name, value):
        ws = workspace.C.Workspace()
        net = core.Net(net_name)
        net.ConstantFill([], [blob_name], shape=[1], value=value)
        ws.run(net)
        self.assertIn(blob_name, ws.blobs)
        self.assertNotIn(net_name, ws.nets)
        np.testing.assert_allclose(
            [value], ws.blobs[blob_name].fetch(), atol=1e-4, rtol=1e-4)

    @given(blob_name=st.text(),
           net_name=st.text(),
           plan_name=st.text(),
           value=st.floats(min_value=-1, max_value=1.0))
    def test_plan_run(self, blob_name, plan_name, net_name, value):
        ws = workspace.C.Workspace()
        plan = core.Plan(plan_name)
        net = core.Net(net_name)
        net.ConstantFill([], [blob_name], shape=[1], value=value)

        plan.AddStep(core.ExecutionStep("step", nets=[net], num_iter=1))

        ws.run(plan)
        self.assertIn(blob_name, ws.blobs)
        self.assertIn(net.Name(), ws.nets)
        np.testing.assert_allclose(
            [value], ws.blobs[blob_name].fetch(), atol=1e-4, rtol=1e-4)

    @given(blob_name=st.text(),
           net_name=st.text(),
           value=st.floats(min_value=-1, max_value=1.0))
    def test_net_create(self, blob_name, net_name, value):
        ws = workspace.C.Workspace()
        net = core.Net(net_name)
        net.ConstantFill([], [blob_name], shape=[1], value=value)
        ws.create_net(net).run()
        self.assertIn(blob_name, ws.blobs)
        self.assertIn(net.Name(), ws.nets)
        np.testing.assert_allclose(
            [value], ws.blobs[blob_name].fetch(), atol=1e-4, rtol=1e-4)

    @given(name=st.text(),
           value=htu.tensor(),
           device_option=st.sampled_from(htu.device_options))
    def test_array_serde(self, name, value, device_option):
        ws = workspace.C.Workspace()
        ws.create_blob(name).feed(value, device_option=device_option)
        self.assertIn(name, ws.blobs)
        blob = ws.blobs[name]
        np.testing.assert_equal(value, ws.blobs[name].fetch())
        serde_blob = ws.create_blob("{}_serde".format(name))
        serde_blob.deserialize(blob.serialize(name))
        np.testing.assert_equal(value, serde_blob.fetch())

    @given(name=st.text(), value=st.text())
    def test_string_serde(self, name, value):
        value = value.encode('ascii', 'ignore')
        ws = workspace.C.Workspace()
        ws.create_blob(name).feed(value)
        self.assertIn(name, ws.blobs)
        blob = ws.blobs[name]
        self.assertEqual(value, ws.blobs[name].fetch())
        serde_blob = ws.create_blob("{}_serde".format(name))
        serde_blob.deserialize(blob.serialize(name))
        self.assertEqual(value, serde_blob.fetch())

    def test_exception(self):
        ws = workspace.C.Workspace()

        with self.assertRaises(TypeError):
            ws.create_net("...")


class TestPredictor(unittest.TestCase):
    def _create_model(self):
        m = model_helper.ModelHelper()
        y = brew.fc(m, "data", "y",
                    dim_in=4, dim_out=2,
                    weight_init=('ConstantFill', dict(value=1.0)),
                    bias_init=('ConstantFill', dict(value=0.0)),
                    axis=0)
        m.net.AddExternalOutput(y)
        return m

    # Use this test with a bigger model to see how using Predictor allows to
    # avoid issues with low protobuf size limit in Python
    #
    # def test_predictor_predefined(self):
    #     workspace.ResetWorkspace()
    #     path = 'caffe2/caffe2/test/assets/'
    #     with open(path + 'squeeze_predict_net.pb') as f:
    #         self.predict_net = f.read()
    #     with open(path + 'squeeze_init_net.pb') as f:
    #         self.init_net = f.read()
    #     self.predictor = workspace.Predictor(self.init_net, self.predict_net)

    #     inputs = [np.zeros((1, 3, 256, 256), dtype='f')]
    #     outputs = self.predictor.run(inputs)
    #     self.assertEqual(len(outputs), 1)
    #     self.assertEqual(outputs[0].shape, (1, 1000, 1, 1))
    #     self.assertAlmostEqual(outputs[0][0][0][0][0], 5.19026289e-05)

    def test_predictor_memory_model(self):
        workspace.ResetWorkspace()
        m = self._create_model()
        workspace.FeedBlob("data", np.zeros([4], dtype='float32'))
        self.predictor = workspace.Predictor(
            workspace.StringifyProto(m.param_init_net.Proto()),
            workspace.StringifyProto(m.net.Proto()))

        inputs = np.array([1, 3, 256, 256], dtype='float32')
        outputs = self.predictor.run([inputs])
        np.testing.assert_array_almost_equal(
            np.array([[516, 516]], dtype='float32'), outputs)


if __name__ == '__main__':
    unittest.main()
