from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def attr_value_proto(dtype, shape, s):
    attr = {}

    if s is not None:
        attr['attr'] = AttrValue(s=s.encode(encoding='utf_8'))

    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr['_output_shapes'] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def tensor_shape_proto(outputsize):
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def node_proto(name,
               op='UnSpecified',
               input=None,
               dtype=None,
               shape=None,  # type: tuple
               outputsize=None,
               attributes=''
               ):
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding='utf_8'),
        op=op,
        input=input,
        attr=attr_value_proto(dtype, outputsize, attributes)
    )
