from caffe2.proto import caffe2_pb2
from caffe.proto import caffe_pb2
from google.protobuf.message import DecodeError, Message
from google.protobuf import text_format
import numpy as np


def CaffeBlobToNumpyArray(blob):
    return np.asarray(blob.data,
                      dtype=np.float32).reshape(
                          blob.num, blob.channels, blob.height, blob.width
                      )


def Caffe2TensorToNumpyArray(tensor):
    return np.asarray(tensor.float_data, dtype=np.float32).reshape(tensor.dims)


def NumpyArrayToCaffe2Tensor(arr, name):
    tensor = caffe2_pb2.TensorProto()
    tensor.data_type = caffe2_pb2.TensorProto.FLOAT
    tensor.name = name
    tensor.dims.extend(arr.shape)
    tensor.float_data.extend(list(arr.flatten().astype(float)))
    return tensor


def MakeArgument(key, value):
    """Makes an argument based on the value type."""
    argument = caffe2_pb2.Argument()
    argument.name = key
    if type(value) is float:
        argument.f = value
    elif type(value) is int or type(value) is bool:
        # We make a relaxation that a boolean variable will also be stored as
        # int.
        argument.i = value
    elif type(value) is str:
        argument.s = value
    elif isinstance(value, Message):
        argument.s = value.SerializeToString()
    elif all(type(v) is float for v in value):
        argument.floats.extend(value)
    elif all(type(v) is int for v in value):
        argument.ints.extend(value)
    elif all(type(v) is str for v in value):
        argument.strings.extend(value)
    elif all(isinstance(v, Message) for v in value):
        argument.strings.extend([v.SerializeToString() for v in value])
    else:
        raise ValueError(
            "Unknown argument type: key=%s value=%s, value type=%s" %
            (key, str(value), str(type(value)))
        )
    return argument


def TryReadProtoWithClass(cls, s):
    """Reads a protobuffer with the given proto class.

    Inputs:
      cls: a protobuffer class.
      s: a string of either binary or text protobuffer content.

    Outputs:
      proto: the protobuffer of cls

    Throws:
      google.protobuf.message.DecodeError: if we cannot decode the message.
    """
    obj = cls()
    try:
        text_format.Parse(s, obj)
        return obj
    except text_format.ParseError as e:
        obj.ParseFromString(s)
        return obj


def GetContentFromProto(obj, function_map):
    """Gets a specific field from a protocol buffer that matches the given class
    """
    for cls, func in function_map.iteritems():
        if type(obj) is cls:
            return func(obj)


def GetContentFromProtoString(s, function_map):
    for cls, func in function_map.iteritems():
        try:
            obj = TryReadProtoWithClass(cls, s)
            return func(obj)
        except DecodeError:
            continue
    else:
        raise DecodeError("Cannot find a fit protobuffer class.")
