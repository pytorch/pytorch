import torch

from typing import Optional, Union
from collections.abc import Sequence
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def attr_value_proto(dtype: object, shape: Optional[Sequence[int]], s: Optional[str]) -> dict[str, AttrValue]:
    """Create a dict of objects matching a NodeDef's attr field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been reverse engineered from
    standard TensorBoard logged data.
    """
    attr = {}
    if s is not None:
        attr["attr"] = AttrValue(s=s.encode(encoding="utf_8"))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr


def tensor_shape_proto(outputsize: Sequence[int]) -> TensorShapeProto:
    """Create an object matching a tensor_shape field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto .
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])


def node_proto(
    name: str,
    op: str = "UnSpecified",
    input: Optional[Union[list[str], str]] = None,
    dtype: Optional[torch.dtype] = None,
    shape: Optional[tuple[int, ...]] = None,
    outputsize: Optional[Sequence[int]] = None,
    attributes: str = "",
) -> NodeDef:
    """Create an object matching a NodeDef.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(
        name=name.encode(encoding="utf_8"),
        op=op,
        input=input,
        attr=attr_value_proto(dtype, outputsize, attributes),
    )
