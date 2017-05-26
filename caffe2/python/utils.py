## @package utils
# Module caffe2.python.utils
from caffe2.proto import caffe2_pb2
from google.protobuf.message import DecodeError, Message
from google.protobuf import text_format
from past.builtins import basestring, long
import collections
import functools
import numpy as np
import sys


def CaffeBlobToNumpyArray(blob):
    if (blob.num != 0):
        # old style caffe blob.
        return (np.asarray(blob.data, dtype=np.float32)
                .reshape(blob.num, blob.channels, blob.height, blob.width))
    else:
        # new style caffe blob.
        return (np.asarray(blob.data, dtype=np.float32)
                .reshape(blob.shape.dim))


def Caffe2TensorToNumpyArray(tensor):
    if tensor.data_type == caffe2_pb2.TensorProto.FLOAT:
        return np.asarray(
            tensor.float_data, dtype=np.float32).reshape(tensor.dims)
    elif tensor.data_type == caffe2_pb2.TensorProto.DOUBLE:
        return np.asarray(
            tensor.double_data, dtype=np.float64).reshape(tensor.dims)
    elif tensor.data_type == caffe2_pb2.TensorProto.INT32:
        return np.asarray(
            tensor.double_data, dtype=np.int).reshape(tensor.dims)
    else:
        # TODO: complete the data type.
        raise RuntimeError(
            "Tensor data type not supported yet: " + str(tensor.data_type))


def NumpyArrayToCaffe2Tensor(arr, name=None):
    tensor = caffe2_pb2.TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name
    if arr.dtype == np.float32:
        tensor.data_type = caffe2_pb2.TensorProto.FLOAT
        tensor.float_data.extend(list(arr.flatten().astype(float)))
    elif arr.dtype == np.float64:
        tensor.data_type = caffe2_pb2.TensorProto.DOUBLE
        tensor.double_data.extend(list(arr.flatten().astype(np.float64)))
    elif arr.dtype == np.int:
        tensor.data_type = caffe2_pb2.TensorProto.INT32
        tensor.int32_data.extend(list(arr.flatten().astype(np.int)))
    else:
        # TODO: complete the data type.
        raise RuntimeError(
            "Numpy data type not supported yet: " + str(arr.dtype))
    return tensor


def MakeArgument(key, value):
    """Makes an argument based on the value type."""
    argument = caffe2_pb2.Argument()
    argument.name = key
    iterable = isinstance(value, collections.Iterable)

    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif isinstance(value, np.generic):
        # convert numpy scalar to native python type
        value = np.asscalar(value)

    if type(value) is float:
        argument.f = value
    elif type(value) is int or type(value) is bool or type(value) is long:
        # We make a relaxation that a boolean variable will also be stored as
        # int.
        argument.i = value
    elif isinstance(value, basestring):
        argument.s = (value if type(value) is bytes
                      else value.encode('utf-8'))
    elif isinstance(value, Message):
        argument.s = value.SerializeToString()
    elif iterable and all(type(v) in [float, np.float_] for v in value):
        argument.floats.extend(value)
    elif iterable and all(type(v) in [int, bool, long, np.int_] for v in value):
        argument.ints.extend(value)
    elif iterable and all(isinstance(v, basestring) for v in value):
        argument.strings.extend([
            (v if type(v) is bytes else v.encode('utf-8')) for v in value])
    elif iterable and all(isinstance(v, Message) for v in value):
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
    except text_format.ParseError:
        obj.ParseFromString(s)
        return obj


def GetContentFromProto(obj, function_map):
    """Gets a specific field from a protocol buffer that matches the given class
    """
    for cls, func in function_map.items():
        if type(obj) is cls:
            return func(obj)


def GetContentFromProtoString(s, function_map):
    for cls, func in function_map.items():
        try:
            obj = TryReadProtoWithClass(cls, s)
            return func(obj)
        except DecodeError:
            continue
    else:
        raise DecodeError("Cannot find a fit protobuffer class.")


def ConvertProtoToBinary(proto_class, filename, out_filename):
    """Convert a text file of the given protobuf class to binary."""
    proto = TryReadProtoWithClass(proto_class, open(filename).read())
    with open(out_filename, 'w') as fid:
        fid.write(proto.SerializeToString())


def GetGPUMemoryUsageStats():
    """Get GPU memory usage stats from CUDAContext. This requires flag
       --caffe2_gpu_memory_tracking to be enabled"""
    from caffe2.python import workspace, core
    workspace.RunOperatorOnce(
        core.CreateOperator(
            "GetGPUMemoryUsage",
            [],
            ["____mem____"],
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0),
        ),
    )
    b = workspace.FetchBlob("____mem____")
    return {
        'total_by_gpu': b[0, :],
        'max_by_gpu': b[1, :],
        'total': np.sum(b[0, :]),
        'max_total': np.sum(b[1, :])
    }


def ResetBlobs(blobs):
    from caffe2.python import workspace, core
    workspace.RunOperatorOnce(
        core.CreateOperator(
            "Free",
            list(blobs),
            list(blobs),
            device_option=core.DeviceOption(caffe2_pb2.CPU),
        ),
    )


class DebugMode(object):
    '''
    This class allows to drop you into an interactive debugger
    if there is an unhandled exception in your python script

    Example of usage:

    def main():
        # your code here
        pass

    if __name__ == '__main__':
        from caffe2.python.utils import DebugMode
        DebugMode.run(main)
    '''

    @classmethod
    def run(cls, func):
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception:
            import pdb

            print(
                'Entering interactive debugger. Type "bt" to print '
                'the full stacktrace. Type "help" to see command listing.')
            print(sys.exc_info()[1])
            print

            pdb.post_mortem()
            sys.exit(1)
            raise


def debug(f):
    '''
    Use this method to decorate your function with DebugMode's functionality

    Example:

    @debug
    def test_foo(self):
        raise Exception("Bar")

    '''

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        def func():
            return f(*args, **kwargs)
        DebugMode.run(func)

    return wrapper
