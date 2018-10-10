from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import convert, workspace
from caffe2.proto import caffe2_pb2, torch_pb2
import unittest
import numpy as np

class TestOperator(unittest.TestCase):
    def setUp(self):
        workspace.ResetWorkspace()

    def testArgument2AttributeProto(self):
        arg_f = caffe2_pb2.Argument()
        arg_f.name = "TestArgF"
        arg_f.f = 10.0
        attr_f = convert.ArgumentToAttributeProto(arg_f)
        self.assertEqual(attr_f.name, arg_f.name)
        self.assertEqual(attr_f.f, arg_f.f)

        arg_i = caffe2_pb2.Argument()
        arg_i.name = "TestArgI"
        arg_i.i = 100
        attr_i = convert.ArgumentToAttributeProto(arg_i)
        self.assertEqual(attr_i.name, arg_i.name)
        self.assertEqual(attr_i.i, arg_i.i)

        arg_s = caffe2_pb2.Argument()
        arg_s.name = "TestArgS"
        arg_s.s = "TestS".encode("utf-8")
        attr_s = convert.ArgumentToAttributeProto(arg_s)
        self.assertEqual(attr_s.name, arg_s.name)
        self.assertEqual(attr_s.s, arg_s.s)

        # TODO: test net arg

        arg_floats = caffe2_pb2.Argument()
        arg_floats.name = "TestArgFloats"
        arg_floats.floats.extend([10.0, 11.0, 12.0])
        attr_floats = convert.ArgumentToAttributeProto(arg_floats)
        self.assertEqual(attr_floats.name, arg_floats.name)
        self.assertEqual(attr_floats.floats, arg_floats.floats)

        arg_ints = caffe2_pb2.Argument()
        arg_ints.name = "TestArgInts"
        arg_ints.ints.extend([100, 101, 102])
        attr_ints = convert.ArgumentToAttributeProto(arg_ints)
        self.assertEqual(attr_ints.name, arg_ints.name)
        self.assertEqual(attr_ints.ints, arg_ints.ints)

        arg_strings = caffe2_pb2.Argument()
        arg_strings.name = "TestArgStrings"
        arg_strings.strings.extend([
            "TestStrings1".encode("utf-8"),
            "TestStrings2".encode("utf-8"),
        ])
        attr_strings = convert.ArgumentToAttributeProto(arg_strings)
        self.assertEqual(attr_strings.name, arg_strings.name)
        self.assertEqual(attr_strings.strings, arg_strings.strings)

        # TODO: test nets arg

    def testAttributeProto2Argument(self):
        attr_f = torch_pb2.AttributeProto()
        attr_f.type = torch_pb2.AttributeProto.FLOAT
        attr_f.name = "TestAttrF"
        attr_f.f = 10.0
        arg_f = convert.AttributeProtoToArgument(attr_f)
        self.assertEqual(arg_f.name, attr_f.name)
        self.assertEqual(arg_f.f, attr_f.f)

        attr_i = torch_pb2.AttributeProto()
        attr_i.type = torch_pb2.AttributeProto.INT
        attr_i.name = "TestArgI"
        attr_i.i = 100
        arg_i = convert.AttributeProtoToArgument(attr_i)
        self.assertEqual(arg_i.name, attr_i.name)
        self.assertEqual(arg_i.i, attr_i.i)

        attr_s = torch_pb2.AttributeProto()
        attr_s.type = torch_pb2.AttributeProto.STRING
        attr_s.name = "TestArgS"
        attr_s.s = "TestS".encode("utf-8")
        arg_s = convert.AttributeProtoToArgument(attr_s)
        self.assertEqual(arg_s.name, attr_s.name)
        self.assertEqual(arg_s.s, attr_s.s)

        # TODO: test graph attribute

        attr_floats = torch_pb2.AttributeProto()
        attr_floats.type = torch_pb2.AttributeProto.FLOATS
        attr_floats.name = "TestAttrFloats"
        attr_floats.floats.extend([10.0, 11.0, 12.0])
        arg_floats = convert.AttributeProtoToArgument(attr_floats)
        self.assertEqual(arg_floats.name, attr_floats.name)
        self.assertEqual(arg_floats.floats, attr_floats.floats)

        attr_ints = torch_pb2.AttributeProto()
        attr_ints.type = torch_pb2.AttributeProto.INTS
        attr_ints.name = "TestArgInts"
        attr_ints.ints.extend([100, 101, 102])
        arg_ints = convert.AttributeProtoToArgument(attr_ints)
        self.assertEqual(arg_ints.name, attr_ints.name)
        self.assertEqual(arg_ints.ints, attr_ints.ints)

        attr_strings = torch_pb2.AttributeProto()
        attr_strings.type = torch_pb2.AttributeProto.STRINGS
        attr_strings.name = "TestArgStrings"
        attr_strings.strings.extend([
            "TestStrings1".encode("utf-8"),
            "TestStrings2".encode("utf-8"),
        ])
        arg_strings = convert.AttributeProtoToArgument(attr_strings)
        self.assertEqual(arg_strings.name, attr_strings.name)
        self.assertEqual(arg_strings.strings, attr_strings.strings)

        # TODO: test graphs attribute


    def testOperatorDef2NodeProto(self):
        op_def = caffe2_pb2.OperatorDef()
        op_def.input.extend(["A", "B", "C"])
        op_def.output.extend(["X", "Y"])
        op_def.name = "TestOpName"
        op_def.type = "TestOp"
        arg1 = caffe2_pb2.Argument()
        arg1.name = "TestArg1"
        arg1.i = 1
        arg2 = caffe2_pb2.Argument()
        arg2.name = "TestArg2"
        arg1.s = "TestInfo".encode("utf-8")
        op_def.arg.extend([arg1, arg2])
        op_def.device_option.CopyFrom(caffe2_pb2.DeviceOption())
        op_def.engine = "TestEngine".encode("utf-8")
        op_def.control_input.extend(["input1", "input2"])
        op_def.is_gradient_op = True
        op_def.debug_info = "TestDebugInfo"

        node = convert.OperatorDefToNodeProto(op_def)

        self.assertEqual(node.input, op_def.input)
        self.assertEqual(node.output, op_def.output)
        self.assertEqual(node.name, op_def.name)
        self.assertEqual(node.op_type, op_def.type)
        self.assertEqual(node.attribute[0].name, op_def.arg[0].name)
        self.assertEqual(node.attribute[1].name, op_def.arg[1].name)
        self.assertEqual(node.device_option, op_def.device_option)
        node_engine = [a.s.decode("utf-8") for a in node.annotations if a.name == "engine"][0]
        self.assertEqual(node_engine, op_def.engine)
        node_control_input = [a.strings for a in node.annotations if a.name == "control_input"][0]
        self.assertEqual(len(node_control_input), len(op_def.control_input))
        for x, y in zip(node_control_input, op_def.control_input):
            self.assertEqual(x.decode("utf-8"), y)
        self.assertEqual(node.doc_string, op_def.debug_info)
        node_is_gradient_op = [a.i for a in node.annotations if a.name == "is_gradient_op"][0]
        self.assertEqual(node_is_gradient_op, int(op_def.is_gradient_op))

    def testNodeProto2OperatorDef(self):
        node = torch_pb2.NodeProto()
        node.input.extend(["A", "B", "C"])
        node.output.extend(["X", "Y"])
        node.name = "TestOpName"
        node.op_type = "TestOp"
        attr1 = torch_pb2.AttributeProto()
        attr1.name = "TestAttr1"
        attr1.type = torch_pb2.AttributeProto.STRING
        attr1.s = "TestInfo".encode("utf-8")
        attr2 = torch_pb2.AttributeProto()
        attr2.name = "TestAttr2"
        attr2.type = torch_pb2.AttributeProto.INT
        attr2.i = 10
        node.attribute.extend([attr1, attr2])
        node.device_option.CopyFrom(caffe2_pb2.DeviceOption())
        anno1 = torch_pb2.AttributeProto()
        anno1.name = "engine"
        anno1.type = torch_pb2.AttributeProto.STRING
        anno1.s = "TestEngine".encode("utf-8")
        anno2 = torch_pb2.AttributeProto()
        anno2.name = "control_input"
        anno2.type = torch_pb2.AttributeProto.STRINGS
        anno2.strings.extend(["input1".encode("utf-8"), "input2".encode("utf-8")])
        anno3 = torch_pb2.AttributeProto()
        anno3.name = "is_gradient_op"
        anno3.type = torch_pb2.AttributeProto.INT
        anno3.i = 1
        node.annotations.extend([anno1, anno2, anno3])
        node.doc_string = "TestDocString".encode("utf-8")

        op_def = convert.NodeProtoToOperatorDef(node)

        self.assertEqual(op_def.input, node.input)
        self.assertEqual(op_def.output, node.output)
        self.assertEqual(op_def.name, node.name)
        self.assertEqual(op_def.type, node.op_type)
        self.assertEqual(op_def.arg[0].name, node.attribute[0].name)
        self.assertEqual(op_def.arg[1].name, node.attribute[1].name)
        self.assertEqual(op_def.device_option, node.device_option)
        node_engine = [a.s for a in node.annotations if a.name == "engine"][0]
        self.assertEqual(op_def.engine, node_engine.decode("utf-8"))
        node_control_input = [a.strings for a in node.annotations if a.name == "control_input"][0]
        for x, y in zip(op_def.control_input, node_control_input):
            self.assertEqual(x, y.decode("utf-8"))
        self.assertEqual(op_def.debug_info, node.doc_string)
        node_is_gradient_op = [a.i for a in node.annotations if a.name == "is_gradient_op"][0]
        self.assertEqual(int(op_def.is_gradient_op), node_is_gradient_op)

    def testEnd2End(self):
        op_def = caffe2_pb2.OperatorDef()
        op_def.type = "Add"
        op_def.input.extend(["input1"])
        op_def.input.extend(["input2"])
        op_def.output.extend(["output1"])
        node = convert.OperatorDefToNodeProto(op_def)

        input1 = np.random.randn(1, 3, 1, 5).astype(np.float32)
        input2 = np.random.randn(2, 1, 4, 1).astype(np.float32)
        ref_output1 = input1 + input2
        workspace.FeedBlob("input1", input1)
        workspace.FeedBlob("input2", input2)
        self.assertEqual(workspace.RunOperatorOnce(node.SerializeToString(), legacy_proto=False), True)

        self.assertEqual(workspace.HasBlob("output1"), True)
        fetched_back = workspace.FetchBlob("output1")
        np.testing.assert_array_equal(fetched_back, ref_output1)

    def testRoundTrip(self):
        op_def = caffe2_pb2.OperatorDef()
        op_def.type = "Add"
        op_def.input.extend(["input1"])
        op_def.input.extend(["input2"])
        op_def.output.extend(["output1"])
        node = convert.OperatorDefToNodeProto(op_def)
        new_op_def = convert.NodeProtoToOperatorDef(node)

        input1 = np.random.randn(1, 3, 1, 5).astype(np.float32)
        input2 = np.random.randn(2, 1, 4, 1).astype(np.float32)
        ref_output1 = input1 + input2
        workspace.FeedBlob("input1", input1)
        workspace.FeedBlob("input2", input2)
        self.assertEqual(workspace.RunOperatorOnce(new_op_def.SerializeToString()), True)

        self.assertEqual(workspace.HasBlob("output1"), True)
        fetched_back = workspace.FetchBlob("output1")
        np.testing.assert_array_equal(fetched_back, ref_output1)


if __name__ == '__main__':
    unittest.main()
