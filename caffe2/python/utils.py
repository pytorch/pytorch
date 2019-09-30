# @package utils
# Module caffe2.python.utils
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python.compatibility import container_abcs
from future.utils import viewitems
from google.protobuf.message import DecodeError, Message
from google.protobuf import text_format

import sys
import copy
import functools
import numpy as np
from six import integer_types, binary_type, text_type, string_types

OPTIMIZER_ITERATION_NAME = "optimizer_iteration"
ITERATION_MUTEX_NAME = "iteration_mutex"


def OpAlmostEqual(op_a, op_b, ignore_fields=None):
    '''
    Two ops are identical except for each field in the `ignore_fields`.
    '''
    ignore_fields = ignore_fields or []
    if not isinstance(ignore_fields, list):
        ignore_fields = [ignore_fields]

    assert all(isinstance(f, text_type) for f in ignore_fields), (
        'Expect each field is text type, but got {}'.format(ignore_fields))

    def clean_op(op):
        op = copy.deepcopy(op)
        for field in ignore_fields:
            if op.HasField(field):
                op.ClearField(field)
        return op

    op_a = clean_op(op_a)
    op_b = clean_op(op_b)
    return op_a == op_b


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
            tensor.int32_data, dtype=np.int).reshape(tensor.dims)   # pb.INT32=>np.int use int32_data
    elif tensor.data_type == caffe2_pb2.TensorProto.INT16:
        return np.asarray(
            tensor.int32_data, dtype=np.int16).reshape(tensor.dims)  # pb.INT16=>np.int16 use int32_data
    elif tensor.data_type == caffe2_pb2.TensorProto.UINT16:
        return np.asarray(
            tensor.int32_data, dtype=np.uint16).reshape(tensor.dims)  # pb.UINT16=>np.uint16 use int32_data
    elif tensor.data_type == caffe2_pb2.TensorProto.INT8:
        return np.asarray(
            tensor.int32_data, dtype=np.int8).reshape(tensor.dims)  # pb.INT8=>np.int8 use int32_data
    elif tensor.data_type == caffe2_pb2.TensorProto.UINT8:
        return np.asarray(
            tensor.int32_data, dtype=np.uint8).reshape(tensor.dims)  # pb.UINT8=>np.uint8 use int32_data
    else:
        # TODO: complete the data type: bool, float16, byte, int64, string
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
    elif arr.dtype == np.int or arr.dtype == np.int32:
        tensor.data_type = caffe2_pb2.TensorProto.INT32
        tensor.int32_data.extend(arr.flatten().astype(np.int).tolist())
    elif arr.dtype == np.int16:
        tensor.data_type = caffe2_pb2.TensorProto.INT16
        tensor.int32_data.extend(list(arr.flatten().astype(np.int16)))  # np.int16=>pb.INT16 use int32_data
    elif arr.dtype == np.uint16:
        tensor.data_type = caffe2_pb2.TensorProto.UINT16
        tensor.int32_data.extend(list(arr.flatten().astype(np.uint16)))  # np.uint16=>pb.UNIT16 use int32_data
    elif arr.dtype == np.int8:
        tensor.data_type = caffe2_pb2.TensorProto.INT8
        tensor.int32_data.extend(list(arr.flatten().astype(np.int8)))   # np.int8=>pb.INT8 use int32_data
    elif arr.dtype == np.uint8:
        tensor.data_type = caffe2_pb2.TensorProto.UINT8
        tensor.int32_data.extend(list(arr.flatten().astype(np.uint8)))   # np.uint8=>pb.UNIT8 use int32_data
    else:
        # TODO: complete the data type: bool, float16, byte, int64, string
        raise RuntimeError(
            "Numpy data type not supported yet: " + str(arr.dtype))
    return tensor


def MakeArgument(key, value):
    """Makes an argument based on the value type."""
    argument = caffe2_pb2.Argument()
    argument.name = key
    iterable = isinstance(value, container_abcs.Iterable)

    # Fast tracking common use case where a float32 array of tensor parameters
    # needs to be serialized.  The entire array is guaranteed to have the same
    # dtype, so no per-element checking necessary and no need to convert each
    # element separately.
    if isinstance(value, np.ndarray) and value.dtype.type is np.float32:
        argument.floats.extend(value.flatten().tolist())
        return argument

    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif isinstance(value, np.generic):
        # convert numpy scalar to native python type
        value = np.asscalar(value)

    if type(value) is float:
        argument.f = value
    elif type(value) in integer_types or type(value) is bool:
        # We make a relaxation that a boolean variable will also be stored as
        # int.
        argument.i = value
    elif isinstance(value, binary_type):
        argument.s = value
    elif isinstance(value, text_type):
        argument.s = value.encode('utf-8')
    elif isinstance(value, caffe2_pb2.NetDef):
        argument.n.CopyFrom(value)
    elif isinstance(value, Message):
        argument.s = value.SerializeToString()
    elif iterable and all(type(v) in [float, np.float_] for v in value):
        argument.floats.extend(
            v.item() if type(v) is np.float_ else v for v in value
        )
    elif iterable and all(
        type(v) in integer_types or type(v) in [bool, np.int_] for v in value
    ):
        argument.ints.extend(
            v.item() if type(v) is np.int_ else v for v in value
        )
    elif iterable and all(
        isinstance(v, binary_type) or isinstance(v, text_type) for v in value
    ):
        argument.strings.extend(
            v.encode('utf-8') if isinstance(v, text_type) else v
            for v in value
        )
    elif iterable and all(isinstance(v, caffe2_pb2.NetDef) for v in value):
        argument.nets.extend(value)
    elif iterable and all(isinstance(v, Message) for v in value):
        argument.strings.extend(v.SerializeToString() for v in value)
    else:
        if iterable:
            raise ValueError(
                "Unknown iterable argument type: key={} value={}, value "
                "type={}[{}]".format(
                    key, value, type(value), set(type(v) for v in value)
                )
            )
        else:
            raise ValueError(
                "Unknown argument type: key={} value={}, value type={}".format(
                    key, value, type(value)
                )
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
    except (text_format.ParseError, UnicodeDecodeError):
        obj.ParseFromString(s)
        return obj


def GetContentFromProto(obj, function_map):
    """Gets a specific field from a protocol buffer that matches the given class
    """
    for cls, func in viewitems(function_map):
        if type(obj) is cls:
            return func(obj)


def GetContentFromProtoString(s, function_map):
    for cls, func in viewitems(function_map):
        try:
            obj = TryReadProtoWithClass(cls, s)
            return func(obj)
        except DecodeError:
            continue
    else:
        raise DecodeError("Cannot find a fit protobuffer class.")


def ConvertProtoToBinary(proto_class, filename, out_filename):
    """Convert a text file of the given protobuf class to binary."""
    with open(filename) as f:
        proto = TryReadProtoWithClass(proto_class, f.read())
    with open(out_filename, 'w') as fid:
        fid.write(proto.SerializeToString())


def GetGPUMemoryUsageStats():
    """Get GPU memory usage stats from CUDAContext/HIPContext. This requires flag
       --caffe2_gpu_memory_tracking to be enabled"""
    from caffe2.python import workspace, core
    workspace.RunOperatorOnce(
        core.CreateOperator(
            "GetGPUMemoryUsage",
            [],
            ["____mem____"],
            device_option=core.DeviceOption(workspace.GpuDeviceType, 0),
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


def raiseIfNotEqual(a, b, msg):
    if a != b:
        raise Exception("{}. {} != {}".format(msg, a, b))


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
        return DebugMode.run(func)

    return wrapper


def BuildUniqueMutexIter(
    init_net,
    net,
    iter=None,
    iter_mutex=None,
    iter_val=0
):
    '''
    Often, a mutex guarded iteration counter is needed. This function creates a
    mutex iter in the net uniquely (if the iter already existing, it does
    nothing)

    This function returns the iter blob
    '''
    iter = iter if iter is not None else OPTIMIZER_ITERATION_NAME
    iter_mutex = iter_mutex if iter_mutex is not None else ITERATION_MUTEX_NAME
    from caffe2.python import core
    if not init_net.BlobIsDefined(iter):
        # Add training operators.
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            iteration = init_net.ConstantFill(
                [],
                iter,
                shape=[1],
                value=iter_val,
                dtype=core.DataType.INT64,
            )
            iter_mutex = init_net.CreateMutex([], [iter_mutex])
            net.AtomicIter([iter_mutex, iteration], [iteration])
    else:
        iteration = init_net.GetBlobRef(iter)
    return iteration


def EnumClassKeyVals(cls):
    # cls can only be derived from object
    assert type(cls) == type
    # Enum attribute keys are all capitalized and values are strings
    enum = {}
    for k in dir(cls):
        if k == k.upper():
            v = getattr(cls, k)
            if isinstance(v, string_types):
                assert v not in enum.values(), (
                    "Failed to resolve {} as Enum: "
                    "duplicate entries {}={}, {}={}".format(
                        cls, k, v, [key for key in enum if enum[key] == v][0], v
                    )
                )
                enum[k] = v
    return enum


def ArgsToDict(args):
    """
    Convert a list of arguments to a name, value dictionary. Assumes that
    each argument has a name. Otherwise, the argument is skipped.
    """
    ans = {}
    for arg in args:
        if not arg.HasField("name"):
            continue
        for d in arg.DESCRIPTOR.fields:
            if d.name == "name":
                continue
            if d.label == d.LABEL_OPTIONAL and arg.HasField(d.name):
                ans[arg.name] = getattr(arg, d.name)
                break
            elif d.label == d.LABEL_REPEATED:
                list_ = getattr(arg, d.name)
                if len(list_) > 0:
                    ans[arg.name] = list_
                    break
        else:
            ans[arg.name] = None
    return ans


def NHWC2NCHW(tensor):
    assert tensor.ndim >= 1
    return tensor.transpose((0, tensor.ndim - 1) + tuple(range(1, tensor.ndim - 1)))


def NCHW2NHWC(tensor):
    assert tensor.ndim >= 2
    return tensor.transpose((0,) + tuple(range(2, tensor.ndim)) + (1,))
