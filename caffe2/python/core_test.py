from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from inspect import currentframe, getframeinfo
import unittest

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, schema, test_util
from caffe2.python.task import Node, Task


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
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
        op = core.CreateOperator("Relu", "x", "y", device_option=device_option)
        self.assertTrue(op.HasField('device_option'))
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        with core.DeviceScope(device_option):
            # from device scope
            op = core.CreateOperator("Relu", "x", "y")
            self.assertTrue(op.HasField('device_option'))
            self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
            self.assertEqual(op.device_option.device_id, 1)
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
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
        with core.DeviceScope(device_option):
            with core.NameScope("foo"):
                op = core.CreateOperator("Relu", "x", "y")
                self.assertTrue(op.HasField('device_option'))
                self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
                self.assertEqual(op.device_option.device_id, 1)
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

    def test_mask_clone_update_external_list(self):
        n = core.Net('original')
        a1 = n.AddExternalInput('a1')
        a2 = n.AddExternalInput('a2')
        p1 = 'p1'
        b1, b2 = n.Concat([a1, a2], ['b1', 'b2'], axis=0)
        c1 = n.Sum([b1, p1], ['c1'])
        c2 = n.Sum([b2], ['c2'])
        n.Sum([c1, c2], ['d'])
        new_net = n.Clone(
            "new", op_id_mask=[0, 1], keep_schema=True, update_external_list=True)
        self.assertEqual(
            sorted(map(str, new_net.external_inputs)),
            ["a1", "a2", "p1"],
            "external input not matched",
        )
        self.assertEqual(
            sorted(map(str, new_net.external_outputs)),
            ["b2", "c1"],
            "external output not matched",
        )
        new_net = n.Clone(
            "new2", op_id_mask=[2, 3], keep_schema=True, update_external_list=True)
        self.assertEqual(
            sorted(map(str, new_net.external_inputs)),
            ["b2", "c1"],
            "external input not matched",
        )
        self.assertEqual(
            sorted(map(str, new_net.external_outputs)),
            ["d"],
            "external output not matched",
        )


class TestExternalInputs(test_util.TestCase):
    def testSetInputRecordWithBlobs(self):
        net = core.Net("test")
        record = schema.NewRecord(net, schema.Struct(
            ("x", schema.Scalar(np.float)),
        ))
        input_record = net.set_input_record(record)
        self.assertTrue(net.BlobIsDefined(input_record.x()))
        self.assertIn(input_record.x(), net.external_inputs)

    def testSetInputRecordWithoutBlobs(self):
        net = core.Net("test")
        record = schema.Struct(("x", schema.Scalar(np.float)))
        input_record = net.set_input_record(record)
        self.assertTrue(net.BlobIsDefined(input_record.x()))
        self.assertIn(input_record.x(), net.external_inputs)


class TestCreateOperator(test_util.TestCase):
    def testCreate(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
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
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertTrue(len(op.arg), 3)

        # can't guarantee ordering of kwargs, so generate a set of args
        # to test with
        arg_map = {}
        for arg in op.arg:
            arg_map[arg.name] = arg

        # Check all elements exist that should
        self.assertEqual("arg1" in arg_map, True)
        self.assertEqual("arg2" in arg_map, True)
        self.assertEqual("arg3" in arg_map, True)

        # Now test that all args were initialized correctly
        self.assertEqual(arg_map["arg1"].i, 1)
        self.assertEqual(arg_map["arg2"].s, b"2")
        self.assertEqual(list(arg_map["arg3"].ints), [1, 2, 3])


class TestAutoNaming(test_util.TestCase):
    def assertOperatorListEqual(self, operatorDefList1, operatorDefList2):
        for op in operatorDefList1:
            op.debug_info = ""
        for op in operatorDefList2:
            op.debug_info = ""
        self.assertEqual(operatorDefList1, operatorDefList2)
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
        self.assertOperatorListEqual(net_a.Proto().op,
                         net_b.Proto().op)
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
            is_test=1,
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


class TestOperatorTraceback(test_util.TestCase):
    def op_name_check(self, net, cf, line, func):
        net.PopulateProtoWithFileName()
        filename = getframeinfo(cf).filename
        self.assertEqual(net.Proto().op[0].name, '{}:{}:{}'.format(
            filename, line, func))

    def test_operator_constructor_traceback(self):
        net = core.Net("test")
        a, b = net.AddExternalInput("a", "b")
        net.Mul([a, b], "c"); cf = currentframe(); line = cf.f_lineno
        func = cf.f_code.co_name
        with self.assertRaises(Exception):
            workspace.RunNetOnce(net)
        with self.assertRaises(Exception):
            workspace.CreateNet(net)
        self.op_name_check(net, cf, line, func)

    def test_operator_runtime_traceback(self):
        net = core.Net("test")
        a = net.AddExternalInput("a")
        workspace.blobs[a] = np.array([1, 2, 3], dtype=np.float32)
        net.Split(a, ["b", "c"], axis=0); cf = currentframe(); line = cf.f_lineno
        func = cf.f_code.co_name
        with self.assertRaises(Exception):
            workspace.RunNetOnce(net)
        workspace.CreateNet(net)
        with self.assertRaises(Exception):
            workspace.RunNet(net)
        self.op_name_check(net, cf, line, func)

    def test_c_workspace_constructor(self):
        net = core.Net("test")
        a, b = net.AddExternalInput("a", "b")
        net.Mul([a, b], "c"); cf = currentframe(); line = cf.f_lineno
        func = cf.f_code.co_name
        ws = workspace.C.Workspace()
        with self.assertRaises(Exception):
            ws.run(net)
        with self.assertRaises(Exception):
            ws.create_net(net)
        self.op_name_check(net, cf, line, func)

    def test_c_workspace_runtime(self):
        net = core.Net("test")
        a = net.AddExternalInput("a")
        net.Split(a, ["b", "c"], axis=0); cf = currentframe(); line = cf.f_lineno
        func = cf.f_code.co_name
        ws = workspace.C.Workspace()
        ws.create_blob(str(a)).feed(np.array([1, 2, 3], dtype=np.float32))
        ws.create_net(net)
        with self.assertRaises(Exception):
            ws.run(net)
        self.op_name_check(net, cf, line, func)

    def test_async_exception_handling(self):
        net = core.Net("test")
        net.Proto().type = 'dag'  # this runs operators on background threads
        a = net.AddExternalInput("a")
        net.Split(a, ["b", "c"], axis=0); cf = currentframe(); line = cf.f_lineno
        func = cf.f_code.co_name
        workspace.FeedBlob(a, np.array([1, 2, 3], dtype=np.float32))
        with self.assertRaises(Exception) as enforceNotMet:
            workspace.RunNetOnce(net)
        self.assertIn('enforce fail', str(enforceNotMet.exception))
        self.op_name_check(net, cf, line, func)


class TestCreatePlan(test_util.TestCase):

    def test_create_plan_from_proto_correctly(self):
        from caffe2.python.net_builder import ops
        with Node('trainer'), Task(name='my_task', num_instances=2) as task:
            with ops.task_init():
                globl = ops.Const(0)
            with ops.task_instance_init():
                local = ops.Const(0)
            with ops.loop(100):
                ops.Copy(globl, local)
            with ops.task_instance_exit():
                ops.Add([globl, local], [globl])
            with ops.task_exit():
                ops.Mul([globl, globl], [globl])

        plan = core.Plan(task.get_step())
        test_plan = core.Plan.create_from_proto(plan.Proto())

        self.assertEqual(len(plan.Steps()), 1)
        self.assertEqual(len(test_plan.Steps()), 1)
        self.assertEqual(len(plan.Proto().network), 9)
        self.assertEqual(len(test_plan.Proto().network), 9)
        self.assertEqual(len(plan.Proto().execution_step), 1)
        self.assertEqual(len(test_plan.Proto().execution_step), 1)
        self.assertEqual(plan.Steps()[0].Name(), test_plan.Steps()[0].Name())
        self.assertEqual(len(plan.Nets()), len(test_plan.Nets()))
        for idx in range(0, len(plan.Nets())):
            # When we create Net for test_plan, we will end up with new Net
            # name with postfix.
            net_1 = plan.Nets()[idx]
            net_2 = test_plan.Nets()[idx]
            trim_size = len(net_1.Name())
            self.assertEqual(net_1.Name(), net_2.Name()[:trim_size])


class TestOpRegistryKey(test_util.TestCase):
    def test_is_operator(self):
        self.assertTrue(core.IsOperator('Relu'))
        self.assertFalse(core.IsOperator('NOEXIST'))

    def test_is_operator_with_engine(self):
        self.assertTrue(core.IsOperatorWithEngine('Relu', 'DEFAULT'))
        self.assertFalse(core.IsOperatorWithEngine('Relu', 'NOEXIST'))


class TestDeviceOption(test_util.TestCase):
    def test_check_equal_node_name(self):
        opt1 = core.DeviceOption(0)
        opt2 = core.DeviceOption(0)
        self.assertTrue(core.device_option_equal(opt1, opt2))
        opt2.node_name = 'test'
        self.assertTrue(core.device_option_equal(opt1, opt2))
        self.assertFalse(core.device_option_equal(opt1, opt2, ignore_node_name=False))
        opt1.node_name = 'test'
        self.assertTrue(core.device_option_equal(opt1, opt2, ignore_node_name=False))

    def test_check_equal_default_value(self):
        opt1 = caffe2_pb2.DeviceOption()
        opt2 = caffe2_pb2.DeviceOption()
        opt1.device_type = 0
        self.assertTrue(core.device_option_equal(opt1, opt2))
        opt1.device_id = 5
        # opt1 still is on CPU, so the options should be equal
        self.assertTrue(core.device_option_equal(opt1, opt2))
        opt2.device_type = 0
        self.assertTrue(core.device_option_equal(opt1, opt2))
        opt1.device_type = 1
        self.assertFalse(core.device_option_equal(opt1, opt2))


class TestInferDeviceCpuOnly(test_util.TestCase):
    def test_inject_copy(self):
        '''
        Test inject cross device copies - this is a no-op on CPU only devices.
        '''
        send_node = 'node:0'
        recv_node = 'node:1'
        # Using placeholder ops for send/recv. Placeholder ops are
        # decorator/fake ops that don't have operator schema.
        placeholder_send = 'Placeholder:Dummy:Send'
        placeholder_recv = 'Placeholder:Dummy:Recv'

        # init_net.
        init_net = core.Net("init_net")
        with core.DeviceScope(0, node_name=send_node):
            init_net.XavierFill([], 'fc_w', shape=[10, 100])
            init_net.ConstantFill([], 'fc_b', shape=[10, ])

        # train_net.
        train_net = core.Net("train_net")
        train_net.Proto().external_input.extend(['fc_w', 'fc_b'])
        with core.DeviceScope(0, node_name=send_node):
            op = core.CreateOperator(
                placeholder_send, ["fc_w", 'fc_b'], [],
                dst_node=recv_node)
            train_net.Proto().op.extend([op])
        with core.DeviceScope(0, node_name=recv_node):
            # Let's rename the recv blob i.e. fc_w -> fc_w_recv.
            op = core.CreateOperator(
                placeholder_recv, [], ['fc_w_recv', 'fc_b'],
                src_node=send_node)
            train_net.Proto().op.extend([op])
            train_net.FC(["data", 'fc_w_recv', 'fc_b'], "fc1")

        # Inject cross device copies.
        init_net, x_dev_state = core.InjectCrossDeviceCopies(
            init_net,
            placeHolderOps=[placeholder_send, placeholder_recv])
        train_net, x_dev_state = core.InjectCrossDeviceCopies(
            train_net, x_dev_state,
            placeHolderOps=[placeholder_send, placeholder_recv])

        # Verify: No Copy operators should be injected since it is CPU only.
        op = train_net.Proto().op[0]
        self.assertEqual(op.type, placeholder_send)
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.input[0], "fc_w")
        self.assertEqual(op.input[1], "fc_b")
        op = train_net.Proto().op[1]
        self.assertEqual(op.type, placeholder_recv)
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.output[0], "fc_w_recv")
        self.assertEqual(op.output[1], "fc_b")
        op = train_net.Proto().op[2]
        self.assertEqual(op.type, "FC")
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.input[1], "fc_w_recv")
        self.assertEqual(op.input[2], "fc_b")


@unittest.skipIf(not workspace.has_gpu_support
                and not workspace.has_hip_support, 'No GPU support')
class TestInferDevice(test_util.TestCase):

    def setUp(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
        self.gpu_option = device_option
        self.cpu_option = caffe2_pb2.DeviceOption()

    def _test_op(
        self,
        op_name,
        in_option,
        out_option,
        op_option=None,
        inputs=None,
        outputs=None
    ):
        op_option = self.gpu_option if not op_option else op_option
        inputs = ["blob_1"] if not inputs else inputs
        outputs = ["blob_2"] if not outputs else outputs
        with core.DeviceScope(op_option):
            op = core.CreateOperator(op_name, inputs, outputs)
        input_dev, output_dev = core.InferOpBlobDevices(op)
        if isinstance(in_option, list):
            assert len(in_option) == len(input_dev), \
                'Length of input device option should match' \
                '{} vs. {}'.format(in_option, input_dev)
            for in_dev, in_opt in zip(input_dev, in_option):
                self.assertEqual(in_dev, in_opt)
        else:
            for in_dev in input_dev:
                self.assertEqual(in_dev, in_option)
        if isinstance(out_option, list):
            assert len(out_option) == len(output_dev), \
                'Length of output device option should match' \
                '{} vs. {}'.format(out_option, output_dev)
            for out_dev, out_opt in zip(output_dev, out_option):
                self.assertEqual(out_dev, out_opt)
        else:
            for out_dev in output_dev:
                self.assertEqual(out_dev, out_option)

    def test_infer_device(self):
        self._test_op(
            "FC",
            self.gpu_option,
            self.gpu_option,
            op_option=self.gpu_option,
            inputs=["data", "fc_w", "fc_b"],
            outputs=["fc_1"]
        )

    def test_infer_device_split_by_lengths(self):
        self._test_op(
            "SplitByLengths",
            [self.gpu_option, self.cpu_option],
            self.gpu_option,
            op_option=self.gpu_option,
            inputs=["data", "fc_w"],
            outputs=["fc_1"]
        )

    def test_infer_device_adam(self):
        in_options = [self.gpu_option] * 6
        in_options[5] = self.cpu_option
        out_options = [self.gpu_option] * 4
        self._test_op(
            "Adam",
            in_options,
            out_options,
            op_option=self.gpu_option,
            inputs=["param", "moment_1", "moment_2", "grad", "lr", "iter"],
            outputs=["output_param", "output_moment_1", "output_moment_2",
                "output_grad"]
        )

    def test_infer_device_cross_device(self):
        self._test_op("CopyGPUToCPU", self.gpu_option, self.cpu_option)
        self._test_op("CopyCPUToGPU", self.cpu_option, self.gpu_option)
        self._test_op("CopyFromCPUInput", self.cpu_option, self.gpu_option)
        self._test_op(
            "CopyFromCPUInput",
            self.cpu_option,
            self.cpu_option,
            op_option=self.cpu_option
        )

    def test_device_inference_function(self):
        # ConcatOp.
        op_option = self.gpu_option
        with core.DeviceScope(op_option):
            op = core.CreateOperator(
                'Concat',
                ['X_{}'.format(i) for i in range(4)],
                ['concat_result', 'split_info'],
                axis=1)
        input_dev, output_dev = core.InferOpBlobDevices(op)
        # 2nd output's type is CPU irrespective of Concat op's device option.
        self.assertEqual(output_dev[1], self.cpu_option)

        #SplitOp.
        op_option = self.gpu_option
        with core.DeviceScope(op_option):
            op = core.CreateOperator(
                'Split',
                ['input', 'split'],
                ['X_{}'.format(i) for i in range(4)],
                axis=0)
        input_dev, output_dev = core.InferOpBlobDevices(op)
        # 2nd input's type is CPU irrespective of Split op's device option.
        self.assertEqual(input_dev[1], self.cpu_option)

    def test_inject_copy(self):
        net = core.Net("test")
        init_net = core.Net("init")
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
        weight = init_net.XavierFill([], 'fc_w', shape=[10, 100])
        bias = init_net.ConstantFill([], 'fc_b', shape=[10, ])

        with core.DeviceScope(device_option):
            net.FC(["data", weight, bias], "fc1")

        _, blob_to_device = core.InjectCrossDeviceCopies(init_net)
        new_net, blob_to_device = core.InjectCrossDeviceCopies(
            net, blob_to_device
        )
        op = new_net._net.op[-1]
        self.assertEqual(op.type, "FC")
        self.assertEqual(op.input[0], "data_gpu_1")
        self.assertEqual(op.input[1], "fc_w_gpu_1")
        self.assertEqual(op.input[2], "fc_b_gpu_1")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(new_net._net.op[-2].type, "CopyCPUToGPU")
        self.assertEqual(new_net._net.op[0].type, "CopyCPUToGPU")
        self.assertNotEqual(blob_to_device["fc_w"], device_option)

    def test_cross_nets(self):
        net = core.Net("test")
        init_net = core.Net("init")
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1
        weight = init_net.XavierFill([], 'fc_w', shape=[10, 100])
        bias = init_net.ConstantFill([], 'fc_b', shape=[10, ])
        const = init_net.ConstantFill([], 'const', shape=[], value=1.)
        with core.DeviceScope(device_option):
            const = init_net.Add([const, const], [const])
            fc_out = net.FC(["data", weight, bias], "fc1")
            net.Add([fc_out, const], [fc_out])

        data_remap = {'data': device_option}
        nets, _ = core.InjectDeviceCopiesAmongNets(
            [init_net, net], blob_to_device_init=data_remap
        )
        op = nets[1]._net.op[0]
        self.assertEqual(op.type, "CopyCPUToGPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.output[0], "fc_w_gpu_1")
        op = nets[1]._net.op[1]
        self.assertEqual(op.type, "CopyCPUToGPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.output[0], "fc_b_gpu_1")
        op = nets[1]._net.op[2]
        self.assertEqual(op.type, "FC")
        self.assertEqual(op.input[0], "data")
        self.assertEqual(op.input[1], "fc_w_gpu_1")
        self.assertEqual(op.input[2], "fc_b_gpu_1")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        op = nets[1]._net.op[3]
        self.assertEqual(op.type, "Add")
        self.assertEqual(op.input[0], "fc1")
        self.assertEqual(op.input[1], "const_gpu_1")
        # check that moved blob is in input to the new net
        for c in ["data", "fc_w", "fc_b", "const_gpu_1"]:
            self.assertTrue(c in nets[1]._net.external_input)
        """
For reference, net.Proto() should be like:
name: ""
op {
  input: "fc_w"
  output: "fc_w_gpu_1"
  name: ""
  type: "CopyCPUToGPU"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "fc_b"
  output: "fc_b_gpu_1"
  name: ""
  type: "CopyCPUToGPU"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "data"
  input: "fc_w_gpu_1"
  input: "fc_b_gpu_1"
  output: "fc1"
  name: ""
  type: "FC"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "fc1"
  input: "const_gpu_1"
  output: "fc1"
  name: ""
  type: "Add"
  device_option {
    device_type: 1
    device_id: 1
  }
}
external_input: "data"
external_input: "fc_w"
external_input: "fc_b"
external_input: "const"
external_input: "const_gpu_1"
"""

    def test_cross_nets_no_change(self):
        net = core.Net("test")
        init_net = core.Net("init")
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1

        with core.DeviceScope(device_option):
            weight = init_net.XavierFill([], 'fc_w', shape=[10, 100])
            bias = init_net.ConstantFill([], 'fc_b', shape=[10, ])
            net.FC(["data", weight, bias], "fc1")

        data_remap = {'data': device_option}
        nets = core.InjectDeviceCopiesAmongNetsWithoutB2D(
            [init_net, net], blob_to_device_init=data_remap
        )
        op = nets[1]._net.op[0]
        self.assertEqual(op.type, "FC")
        self.assertEqual(op.input[0], "data")
        self.assertEqual(op.input[1], "fc_w")
        self.assertEqual(op.input[2], "fc_b")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        """
For reference, net.Proto() should be like:
name: ""
op {
  input: "data"
  input: "fc_w"
  input: "fc_b"
  output: "fc1"
  name: ""
  type: "FC"
  device_option {
    device_type: 1
    device_id: 1
  }
}
external_input: "data"
external_input: "fc_w"
external_input: "fc_b"
"""

    def test_inject_copy_multi_use(self):
        net = core.Net("test")
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1

        with core.DeviceScope(device_option):
            net.Relu("data", "relu1")
        net.Relu("data", "relu2")
        with core.DeviceScope(device_option):
            net.Relu("data", "relu3")
        net.Relu("data", "relu4")
        device_option.device_id = 0
        with core.DeviceScope(device_option):
            net.Relu("data", "relu5")
        device_option.device_id = 1
        with core.DeviceScope(device_option):
            net.Relu("data", "relu6")

        new_net, _ = core.InjectCrossDeviceCopies(net)
        op = new_net._net.op[0]
        self.assertEqual(op.type, "CopyCPUToGPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.output[0], "data_gpu_1")
        op = new_net._net.op[1]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.output[0], "relu1")
        op = new_net._net.op[2]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.output[0], "relu2")
        op = new_net._net.op[3]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.input[0], "data_gpu_1")
        self.assertEqual(op.output[0], "relu3")
        op = new_net._net.op[4]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.output[0], "relu4")
        op = new_net._net.op[5]
        self.assertEqual(op.type, "CopyCPUToGPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 0)
        self.assertEqual(op.output[0], "data_gpu_0")
        op = new_net._net.op[6]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 0)
        self.assertEqual(op.input[0], "data_gpu_0")
        self.assertEqual(op.output[0], "relu5")
        op = new_net._net.op[7]
        self.assertEqual(op.type, "Relu")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 1)
        self.assertEqual(op.input[0], "data_gpu_1")
        self.assertEqual(op.output[0], "relu6")
        """
For reference, net.Proto() should be like:
name: ""
op {
  input: "data"
  output: "data_gpu_1"
  name: ""
  type: "CopyCPUToGPU"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "data_gpu_1"
  output: "relu1"
  name: ""
  type: "Relu"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "data"
  output: "relu2"
  name: ""
  type: "Relu"
}
op {
  input: "data_gpu_1"
  output: "relu3"
  name: ""
  type: "Relu"
  device_option {
    device_type: 1
    device_id: 1
  }
}
op {
  input: "data"
  output: "relu4"
  name: ""
  type: "Relu"
}
op {
  input: "data"
  output: "data_gpu_0"
  name: ""
  type: "CopyCPUToGPU"
  device_option {
    device_type: 1
    device_id: 0
  }
}
op {
  input: "data_gpu_0"
  output: "relu5"
  name: ""
  type: "Relu"
  device_option {
    device_type: 1
    device_id: 0
  }
}
op {
  input: "data_gpu_1"
  output: "relu6"
  name: ""
  type: "Relu"
  device_option {
    device_type: 1
    device_id: 1
  }
}
external_input: "data"
"""

    def test_inject_copy_placeholder_ops(self):
        '''
        Test inject cross device copies with placeholder ops. Placeholder ops
        are decorator/fake ops that don't have operator schema.
        '''
        # Create CPU and GPU devices on 2 nodes.
        cpu_device = []
        gpu_device = []
        for i in range(0, 2):
            cpu_device.append(caffe2_pb2.DeviceOption())
            cpu_device[i].node_name = 'node:' + str(i)
            gpu_device.append(caffe2_pb2.DeviceOption())
            gpu_device[i].device_type = workspace.GpuDeviceType
            gpu_device[i].device_id = 0
            gpu_device[i].node_name = 'node:' + str(i)
        send_node = 'node:0'
        recv_node = 'node:1'
        placeholder_send = 'Placeholder:Dummy:Send'
        placeholder_recv = 'Placeholder:Dummy:Recv'

        # init_net.
        init_net = core.Net("init_net")
        with core.DeviceScope(gpu_device[0]):
            weight = init_net.XavierFill([], 'fc_w', shape=[10, 100])
            bias = init_net.ConstantFill([], 'fc_b', shape=[10, ])
        with core.DeviceScope(cpu_device[0]):
            op = core.CreateOperator(
                placeholder_send, [weight, bias], [],
                dst_node=recv_node)
            init_net._net.op.extend([op])

        # train_net
        train_net = core.Net("train_net")
        with core.DeviceScope(cpu_device[1]):
            # XXX. replace hardcoded op name. Move test to net_transforms.
            op = core.CreateOperator(
                placeholder_recv, [], [weight, bias],
                src_node=send_node)
            train_net._net.op.extend([op])
            train_net.FC(["data", weight, bias], "fc1")

        # Inject cross device copies.
        init_net, x_dev_state = core.InjectCrossDeviceCopies(
            init_net,
            placeHolderOps=[placeholder_send, placeholder_recv])
        train_net, x_dev_state = core.InjectCrossDeviceCopies(
            train_net, x_dev_state,
            placeHolderOps=[placeholder_send, placeholder_recv])

        # Verify (init_net)
        op = init_net._net.op[2]
        self.assertEqual(op.type, "CopyGPUToCPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 0)
        self.assertEqual(op.output[0], "fc_w_cpu")
        op = init_net._net.op[3]
        self.assertEqual(op.type, "CopyGPUToCPU")
        self.assertEqual(op.device_option.device_type, workspace.GpuDeviceType)
        self.assertEqual(op.device_option.device_id, 0)
        self.assertEqual(op.output[0], "fc_b_cpu")
        op = init_net._net.op[4]
        self.assertEqual(op.type, placeholder_send)
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.input[0], "fc_w_cpu")
        self.assertEqual(op.input[1], "fc_b_cpu")
        # Verify (train_net)
        op = train_net._net.op[0]
        self.assertEqual(op.type, placeholder_recv)
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.output[0], "fc_w_cpu")
        self.assertEqual(op.output[1], "fc_b_cpu")
        op = train_net._net.op[3]
        self.assertEqual(op.type, "FC")
        self.assertEqual(op.device_option.device_type, 0)
        self.assertEqual(op.input[1], "fc_w_cpu")
        self.assertEqual(op.input[2], "fc_b_cpu")

    def test_blob_inplace(self):
        net = core.Net("test")
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = 1

        net.Adagrad(['param', 'moment', 'grad', 'lr'], ['param', 'moment'])
        with core.DeviceScope(device_option):
            net.Relu("param", "param_relu_no_sense")
        net, _ = core.InjectCrossDeviceCopies(net)
        op = net._net.op[1]
        self.assertEqual(op.type, 'CopyCPUToGPU')
        self.assertEqual(op.input[0], 'param')
        self.assertEqual(op.output[0], 'param_gpu_1')
        op = net._net.op[2]
        self.assertEqual(op.input[0], 'param_gpu_1')

        net.Relu('nonsense_input', 'moment')
        # should not raise inplace error
        core.InjectCrossDeviceCopies(net)
        with core.DeviceScope(device_option):
            net.Relu('nonsense_input_gpu', 'moment')
        with self.assertRaises(RuntimeError):
            core.InjectCrossDeviceCopies(net)


class TestRerouteTensor(test_util.TestCase):
    def test_reroute_tensor(self):
        net = core.Net("reroute_tensor")
        net.Conv(["input", "w", "b"], "conv1")
        net.Relu(["conv1"], "conv1_relu")
        new_op = core.CreateOperator("SpatialBN",
            ["conv1", "scale", "bias", "mean", "var"],
            ["conv1_bn", "mean", "var", "saved_mean", "saved_var"])
        # insert bn between conv and relu
        net.reroute_tensor("conv1", new_op, [net.Proto().op[1]])
        self.assertEqual(new_op, net.Proto().op[1], "insertion failed")
        self.assertEqual(net.Proto().op[2].input[0], "conv1_bn", "reroute failed")


if __name__ == '__main__':
    unittest.main()
