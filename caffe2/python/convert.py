## @package workspace
# Module caffe2.python.workspace
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2, torch_pb2

import caffe2.python._import_c_extension as C


def ArgumentToAttributeProto(arg):
    serialized_arg = None
    if hasattr(arg, 'SerializeToString') and callable(arg.SerializeToString):
        serialized_arg = arg.SerializeToString()
    elif isinstance(arg, bytes):
        serialized_arg = arg
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither arg is bytes.\ntype is {}'.format(type(arg)))
    attr = torch_pb2.AttributeProto()
    attr.ParseFromString(C.argument_to_attribute_proto(serialized_arg))
    return attr


def AttributeProtoToArgument(attr):
    serialized_attr = None
    if hasattr(attr, 'SerializeToString') and callable(attr.SerializeToString):
        serialized_attr = attr.SerializeToString()
    elif isinstance(attr, bytes):
        serialized_attr = attr
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither attr is bytes.\ntype is {}'.format(type(attr)))
    arg = caffe2_pb2.Argument()
    arg.ParseFromString(C.attribute_proto_to_argument(serialized_attr))
    return arg


def OperatorDefToNodeProto(op_def):
    serialized_op_def = None
    if hasattr(op_def, 'SerializeToString') and callable(op_def.SerializeToString):
        serialized_op_def = op_def.SerializeToString()
    elif isinstance(op_def, bytes):
        serialized_op_def = op_def
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither op_def is bytes.\ntype is {}'.format(type(op_def)))
    node = torch_pb2.NodeProto()
    node.ParseFromString(C.operator_def_to_node_proto(serialized_op_def))
    return node


def NodeProtoToOperatorDef(node_proto):
    serialized_node_proto = None
    if hasattr(node_proto, 'SerializeToString') and callable(node_proto.SerializeToString):
        serialized_node_proto = node_proto.SerializeToString()
    elif isinstance(node_proto, bytes):
        serialized_node_proto = node_proto
    else:
        raise ValueError('No SerializeToString method is detected. '
                         'neither node_proto is bytes.\ntype is {}'.format(type(node_proto)))
    op_def = caffe2_pb2.OperatorDef()
    op_def.ParseFromString(C.node_proto_to_operator_def(serialized_node_proto))
    return op_def
