import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, test_util


class TestScopes(test_util.TestCase):
    def testBlobReferenceIsIndependentFromNameScope(self):
        blob_v = core.BlobReference("v")
        with core.NameScope("foo"):
            blob_w = core.BlobReference("w")
            with core.NameScope("bar"):
                blob_x = core.BlobReference("x")
        self.assertEqual(str(blob_v), "v")
        self.assertEqual(str(blob_w), "w")
        self.assertEqual(str(blob_x), "x")

    def testNameScopeWithOp(self):
        global_x = core.BlobReference("x")
        global_y = core.BlobReference("y")
        with core.NameScope("foo"):
            # Raw strings should have namescope prepended.
            op = core.CreateOperator("Relu", "x", "y")
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "foo/x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "foo/y")
            # BlobReferences should not.
            op = core.CreateOperator("Relu", global_x, global_y)
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "y")

    def testNameScopeWithReset(self):
        with core.NameScope("foo"):
            # foo/
            op = core.CreateOperator("Relu", "x", "y")
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "foo/x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "foo/y")
            with core.NameScope("bar"):
                # foo/bar/
                op = core.CreateOperator("Relu", "x", "y")
                self.assertEqual(len(op.input), 1)
                self.assertEqual(op.input[0], "foo/bar/x")
                self.assertEqual(len(op.output), 1)
                self.assertEqual(op.output[0], "foo/bar/y")
            # Back to foo/
            op = core.CreateOperator("Relu", "x", "y")
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "foo/x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "foo/y")
            with core.NameScope("bar", reset=True):
                # bar/
                op = core.CreateOperator("Relu", "x", "y")
                self.assertEqual(len(op.input), 1)
                self.assertEqual(op.input[0], "bar/x")
                self.assertEqual(len(op.output), 1)
                self.assertEqual(op.output[0], "bar/y")
            # Back to foo/
            op = core.CreateOperator("Relu", "x", "y")
            self.assertEqual(len(op.input), 1)
            self.assertEqual(op.input[0], "foo/x")
            self.assertEqual(len(op.output), 1)
            self.assertEqual(op.output[0], "foo/y")

    def testDeviceScope(self):
        # No device
        op = core.CreateOperator("Relu", "x", "y")
        self.assertFalse(op.HasField('device_option'))
        # explicitly setting a device
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = 1
        op = core.CreateOperator("Relu", "x", "y", device_option=device_option)
        self.assertTrue(op.HasField('device_option'))
        self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
        self.assertEqual(op.device_option.cuda_gpu_id, 1)
        with core.DeviceScope(device_option):
            # from device scope
            op = core.CreateOperator("Relu", "x", "y")
            self.assertTrue(op.HasField('device_option'))
            self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
            self.assertEqual(op.device_option.cuda_gpu_id, 1)
            # from an overridden device option
            override_device = caffe2_pb2.DeviceOption()
            override_device.device_type = caffe2_pb2.CPU
            op = core.CreateOperator(
                "Relu", "x", "y", device_option=override_device)
            self.assertTrue(op.HasField('device_option'))
            self.assertEqual(op.device_option.device_type, caffe2_pb2.CPU)
        # back from normal: no device
        op = core.CreateOperator("Relu", "x", "y")
        self.assertFalse(op.HasField('device_option'))
        device_option = caffe2_pb2.DeviceOption()

    def testNameAndDeviceScopeTogether(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = 1
        with core.DeviceScope(device_option):
            with core.NameScope("foo"):
                op = core.CreateOperator("Relu", "x", "y")
                self.assertTrue(op.HasField('device_option'))
                self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
                self.assertEqual(op.device_option.cuda_gpu_id, 1)
                self.assertEqual(len(op.input), 1)
                self.assertEqual(op.input[0], "foo/x")
                self.assertEqual(len(op.output), 1)
                self.assertEqual(op.output[0], "foo/y")


class TestCloneNet(test_util.TestCase):
    def testPartialClone(self):
        params = core.Net('params')
        p1 = params.ConstantFill([], ['p1'])
        workspace.CreateNet(params)
        workspace.RunNetOnce(params)

        n = core.Net('original')
        a1 = n.AddExternalInput('a1')
        a2 = n.AddExternalInput('a2')
        b1, b2 = n.Concat([a1, a2], ['b1', 'b2'], axis=0)
        c1 = n.Sum([b1, p1], ['c1'])
        c2 = n.Sum([b2], ['c2'])
        d = n.Sum([c1, c2], ['d'])

        # test that gradient ops are ignored when partial-cloning
        n.AddGradientOperators([d])

        # test some in-place ops
        k = n.Sum([p1], ['k'])
        e = n.Sum([d], ['e'])
        e = n.Sum([e, k], [e])
        e = n.Sum([e], [e])
        f = n.Sum(e, ['f'])

        def net_assert(net, num_ops, inputs, outputs, internals):
            self.assertEqual(len(net.Proto().op), num_ops)
            self.assertEqual(set(net.Proto().external_input), inputs)
            self.assertEqual(set(net.Proto().external_output), outputs)
            all_blobs = set(net.Proto().external_input)
            all_blobs |= set(net.Proto().external_output)
            for op in net.Proto().op:
                all_blobs |= set(op.input) | set(op.output)
            self.assertEqual(all_blobs, inputs | outputs | internals)
            # create net to make sure its valid
            for input in inputs:
                workspace.FeedBlob(input, np.array([]))
            workspace.CreateNet(net)

        n2, (d22, ) = n.ClonePartial('f1', {a1: 'a11', a2: 'a22'}, [d])
        net_assert(
            n2, 4, {'p1', 'a11', 'a22'}, {'f1/d'},
            {'f1/b1', 'f1/b2', 'f1/c1', 'f1/c2', 'p1'})
        self.assertTrue(isinstance(d22, core.BlobReference))
        self.assertEqual(d22.Net(), n2)
        self.assertEqual(str(d22), 'f1/d')

        n3, (d22, ) = n.ClonePartial('f2', [b1, b2], [d])
        net_assert(
            n3, 3, {'p1', 'b1', 'b2'}, {'f2/d'}, {'f2/c1', 'f2/c2', 'p1'})
        self.assertEqual(str(d22), 'f2/d')

        n4, (c22, ) = n.ClonePartial('f3', [b1], [c1])
        net_assert(n4, 1, {'p1', 'b1'}, {'f3/c1'}, {'p1'})
        self.assertEqual(str(c22), 'f3/c1')

        n5, (c11, c22) = n.ClonePartial('f4', [b1, b2], [c1, c2])
        net_assert(n5, 2, {'p1', 'b1', 'b2'}, {'f4/c1', 'f4/c2'}, {'p1'})
        self.assertEqual(str(c11), 'f4/c1')
        self.assertEqual(str(c22), 'f4/c2')

        with self.assertRaises(AssertionError):
            n.ClonePartial('f4', [a1, a2, c2], [d])

        n6, (e22, ) = n.ClonePartial('f5', [d], [e])
        net_assert(n6, 4, {'p1', 'd'}, {'f5/e'}, {'f5/k', 'p1'})
        self.assertEqual(str(e22), 'f5/e')

        n8, (e22, f22) = n.ClonePartial('f7', [d], [e, f])
        net_assert(n8, 5, {'p1', 'd'}, {'f7/e', 'f7/f'}, {'p1', 'f7/k'})
        self.assertEqual(str(e22), 'f7/e')
        self.assertEqual(str(f22), 'f7/f')

        params._CheckLookupTables()
        n._CheckLookupTables()


class TestCreateOperator(test_util.TestCase):
    def testCreate(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = 1
        op = core.CreateOperator(
            "Ludicrous", "x", "y", name="ludicrous",
            control_input="z", device_option=device_option,
            engine="WARP", arg1=1, arg2="2", arg3=[1, 2, 3])
        self.assertEqual(op.type, "Ludicrous")
        self.assertEqual(op.name, "ludicrous")
        self.assertEqual(op.engine, "WARP")
        self.assertEqual(len(op.input), 1)
        self.assertEqual(op.input[0], "x")
        self.assertEqual(len(op.output), 1)
        self.assertEqual(op.output[0], "y")
        self.assertEqual(len(op.control_input), 1)
        self.assertEqual(op.control_input[0], "z")
        self.assertTrue(op.HasField('device_option'))
        self.assertEqual(op.device_option.device_type, caffe2_pb2.CUDA)
        self.assertEqual(op.device_option.cuda_gpu_id, 1)
        self.assertTrue(len(op.arg), 3)
        self.assertEqual(op.arg[0].name, "arg1")
        self.assertEqual(op.arg[1].name, "arg2")
        self.assertEqual(op.arg[2].name, "arg3")
        self.assertEqual(op.arg[0].i, 1)
        self.assertEqual(op.arg[1].s, b"2")
        self.assertEqual(list(op.arg[2].ints), [1, 2, 3])

    def testCreateWithNoneKwarg(self):
        with self.assertRaises(ValueError):
            core.CreateOperator("Ludicrous", "x", "y", arg1=None)


class TestAutoNaming(test_util.TestCase):
    """
    Test that operators are named with different names, and that automatically
    named blob names don't clash intra or inter networks.
    """
    def test_next_blob(self):
        def create_net():
            net = core.Net('net')
            with core.NameScope('foo'):
                net.Add(['a', 'b'], net.NextScopedBlob('ab'))

            net.Add(['c', 'd'], net.NextBlob('cd'))
            return net

        net_a = create_net()
        net_b = create_net()
        # created net proto is predicatable.
        self.assertEqual(net_a.Proto().op, net_b.Proto().op)
        self.assertEqual(net_a.Proto().op[0].output[0], 'foo/ab')
        self.assertEqual(net_a.Proto().op[1].output[0], 'cd')

        net_c = core.Net('net')
        # different calls return different blob names
        self.assertNotEqual(str(net_c.NextBlob('b')), str(net_c.NextBlob('b')))

    def test_auto_naming(self):
        a = core.Net('net')
        b = core.Net('net')
        self.assertNotEqual(a.Proto().name, b.Proto().name)
        a_in1 = a.AddExternalInput('a')
        b_in1 = b.AddExternalInput('b')
        all_outputs_single = []
        all_outputs_list = []

        def add_ops():
            all_outputs_single.append(a.Sum([a_in1, a_in1]))
            all_outputs_single.append(a.Sum([a_in1, a_in1]))
            all_outputs_single.append(b.Sum([b_in1, b_in1]))
            all_outputs_single.append(b.Sum([b_in1, b_in1]))
            all_outputs_list.append(a.Sum([a_in1, a_in1], outputs=2))
            all_outputs_list.append(a.Sum([a_in1, a_in1], outputs=2))
            all_outputs_list.append(b.Sum([b_in1, b_in1], outputs=2))
            all_outputs_list.append(b.Sum([b_in1, b_in1], outputs=2))

        add_ops()
        with core.NameScope('n1'):
            add_ops()

        # Force reset of lookup tables
        a.Proto().name

        with core.NameScope('n2'):
            add_ops()

        all_outputs = []
        for s in all_outputs_single:
            all_outputs.append(str(s))
        for l in all_outputs_list:
            for o in l:
                all_outputs.append(str(o))

        for i, o1 in enumerate(all_outputs):
            for j, o2 in enumerate(all_outputs):
                if i != j:
                    self.assertNotEqual(str(o1), str(o2))

        a._CheckLookupTables()
        b._CheckLookupTables()


class TestAppendNet(test_util.TestCase):

    def test_external_inputs_merged_correctly(self):
        netA = core.Net("A")
        netA.Sum(["in1", "in2"], ["sum1"])
        self.assertTrue("in1" in netA.external_inputs)

        netB = core.Net("B")
        netB.Sum(["in3", "in4"], ["in1"])
        netB.AppendNet(netA)
        self.assertFalse("in1" in netB.external_inputs)

    def test_external_inputs_merged_correctlyB(self):
        netA = core.Net("A")
        netA.Sum(["in1", "in2"], ["sum1"])
        self.assertTrue("in1" in netA.external_inputs)

        netB = core.Net("B")
        netB.Sum(["in3", "in4"], ["in1"])
        netA.AppendNet(netB)  # note different order than in prev test
        self.assertTrue("in1" in netA.external_inputs)


class TestExtractPredictorNet(test_util.TestCase):

    def test_extract_simple(self):
        from caffe2.python import brew
        from caffe2.python.model_helper import ModelHelper, ExtractPredictorNet

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        [data, label] = brew.image_input(
            model,
            "reader", ["xx/data", "label"],
        )
        cnv = brew.conv(model, data, 'cnv', 32, 32, 4)
        a = brew.fc(model, cnv, 'a', 100, 200)
        pred = brew.fc(model, a, 'pred', 200, 5)
        brew.softmax(model, [pred, label], "softmax")

        (predict_net, export_blobs) = ExtractPredictorNet(
            net_proto=model.net.Proto(),
            input_blobs=["xx/data"],
            output_blobs=["pred"],
            renames={"xx/data": "image"},
        )
        export_blobs = set(export_blobs)

        ops = list(predict_net.Proto().op)
        for op in ops:
            self.assertFalse(op.type == "Softmax")
            self.assertFalse("xx/data" in op.input)

        # Note: image input should not be included
        self.assertEquals(ops[0].type, "Conv")
        self.assertEquals(ops[1].type, "FC")
        self.assertEquals(ops[2].type, "FC")
        self.assertEquals(len(ops), 3)

        # test rename happened
        self.assertEquals(ops[0].input[0], "image")

        # Check export blobs
        self.assertTrue("image" not in export_blobs)
        self.assertTrue("xx/data" not in export_blobs)
        self.assertEqual(set([str(p) for p in model.params]), export_blobs)

        # Check external inputs/outputs
        self.assertTrue("image" in predict_net.Proto().external_input)
        self.assertEquals(set(["pred"]), set(predict_net.Proto().external_output))
        self.assertEqual(
            set(predict_net.Proto().external_input) -
            set([str(p) for p in model.params]), set(["image"])
        )


if __name__ == '__main__':
    unittest.main()
