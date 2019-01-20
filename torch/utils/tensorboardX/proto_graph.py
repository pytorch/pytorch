from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue
from .proto.tensor_shape_pb2 import TensorShapeProto

from collections import defaultdict

# nodes.append(
#     NodeDef(name=node['name'], op=node['op'], input=node['inputs'],
#             attr={'lanpa': AttrValue(s=node['attr'].encode(encoding='utf_8')),
#                   '_output_shapes': AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))}))


def AttrValue_proto(dtype,
                    shape,
                    s,
                    ):
    attr = {}

    if s is not None:
        attr['attr'] = AttrValue(s=s.encode(encoding='utf_8'))

    if shape is not None:
        shapeproto = TensorShape_proto(shape)
        attr['_output_shapes'] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def TensorShape_proto(outputsize):
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def Node_proto(name,
               op='UnSpecified',
               input=[],
               dtype=None,
               shape=None,  # type: tuple
               outputsize=None,
               attributes=''
               ):
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding='utf_8'),
        op=op,
        input=input,
        attr=AttrValue_proto(dtype, outputsize, attributes)
    )
