import os
import ctypes
import warnings
import torch.cuda
from torch.backends.cudnn import int_array

lib = None

__all__ = ['all_reduce', 'reduce', 'broadcast', 'all_gather', 'reduce_scatter']


def _libnccl():
    global lib
    if lib is None:
        lib = ctypes.pydll.LoadLibrary(None)
        if hasattr(lib, 'ncclCommDestroy'):
            lib.ncclCommDestroy.restype = None
        else:
            lib = None
    return lib


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    if _libnccl() is None:
        warnings.warn('NCCL library not found. Check your LD_LIBRARY_PATH')
        return False

    return True


_communicators = {}

# ncclDataType_t
ncclChar = 0
ncclInt = 1
ncclHalf = 2
ncclFloat = 3
ncclDouble = 4
ncclInt64 = 5
ncclUint64 = 6

# ncclRedOp_t
SUM = 0
PROD = 1
MAX = 2
MIN = 3

status_codes = {
    0: "Success",
    1: "Unhandled Cuda Error",
    2: "System Error",
    3: "Internal Error",
    4: "Invalid Device Pointer",
    5: "Invalid Rank",
    6: "Unsupported Device Count",
    7: "Device Not Found",
    8: "Invalid Device Index",
    9: "Lib Wrapper Not Set",
    10: "Cuda Malloc Failed",
    11: "Rank Mismatch",
    12: "Invalid Argument",
    13: "Invalid Type",
    14: "Invalid Operation",
}

nccl_types = {
    'torch.cuda.ByteTensor': ncclChar,
    'torch.cuda.CharTensor': ncclChar,
    'torch.cuda.IntTensor': ncclInt,
    'torch.cuda.HalfTensor': ncclHalf,
    'torch.cuda.FloatTensor': ncclFloat,
    'torch.cuda.DoubleTensor': ncclDouble,
    'torch.cuda.LongTensor': ncclInt64,
}


class NcclError(RuntimeError):

    def __init__(self, status):
        self.status = status
        msg = '{0} ({1})'.format(status_codes.get(status), status)
        super(NcclError, self).__init__(msg)


class NcclComm(ctypes.c_void_p):
    pass


class NcclCommList(object):

    def __init__(self, devices):
        self.devices = devices
        ptrs = (NcclComm * len(devices))()
        self._as_parameter_ = ptrs
        check_error(lib.ncclCommInitAll(self, len(devices), int_array(devices)))

    def __getitem__(self, i):
        return self._as_parameter_[i]

    def __del__(self):
        for i in range(len(self.devices)):
            lib.ncclCommDestroy(self[i])


def check_error(status):
    if status != 0:
        raise NcclError(status)


def communicator(inputs, outputs=None):
    if _libnccl() is None:
        raise RuntimeError('Unable to load NCCL library')

    devices = [input.get_device() for input in inputs]
    if outputs is not None:
        for device, output in zip(devices, outputs):
            if output.get_device() != device:
                raise ValueError("inputs and outputs must be on the same devices")

    key = ','.join(str(d) for d in devices)
    if key not in _communicators:
        _communicators[key] = NcclCommList(devices)

    return _communicators[key]


def cudaStream():
    # TODO: return the current stream
    # ffi.C.THCState_getCurrentStream(cutorch.getState())
    return None


def all_reduce(inputs, outputs=None, op=SUM):
    if outputs is None:
        outputs = inputs
    _check_inputs(inputs, outputs)
    comm = communicator(inputs, outputs)
    count = inputs[0].numel()
    data_type = nccl_types[inputs[0].type()]
    with torch.cuda._free_mutex():
        for i in range(len(inputs)):
            with torch.cuda.device(comm.devices[i]):
                check_error(lib.ncclAllReduce(
                    ctypes.c_void_p(inputs[i].data_ptr()),
                    ctypes.c_void_p(outputs[i].data_ptr()),
                    count, data_type, op, comm[i], cudaStream()))


def reduce(inputs, outputs=None, root=0, op=SUM, streams=None):
    assert(root >= 0 and root < len(inputs))
    if outputs is None:
        outputs = inputs
    if streams is None:
        streams = [None] * len(inputs)
    _check_inputs(inputs, outputs)
    comm = communicator(inputs)
    count = inputs[0].numel()
    data_type = nccl_types[inputs[0].type()]
    with torch.cuda._free_mutex():
        for i in range(len(inputs)):
            with torch.cuda.device(comm.devices[i]):
                check_error(lib.ncclReduce(
                    ctypes.c_void_p(inputs[i].data_ptr()),
                    ctypes.c_void_p(outputs[i].data_ptr()), count,
                    data_type, op, root, comm[i], streams[i]))


def broadcast(inputs, root=0):
    assert(root >= 0 and root < len(inputs))
    _check_inputs(inputs, inputs)
    comm = communicator(inputs)
    count = inputs[0].numel()
    data_type = nccl_types[inputs[0].type()]
    with torch.cuda._free_mutex():
        for i in range(len(inputs)):
            with torch.cuda.device(comm.devices[i]):
                check_error(lib.ncclBcast(
                    ctypes.c_void_p(inputs[i].data_ptr()), count,
                    data_type, root, comm[i], cudaStream()))


def all_gather(inputs, outputs):
    _check_inputs(inputs, outputs, len(inputs))
    comm = communicator(inputs, outputs)
    count = inputs[0].numel()
    data_type = nccl_types[inputs[0].type()]
    with torch.cuda._free_mutex():
        for i in range(len(inputs)):
            with torch.cuda.device(comm.devices[i]):
                check_error(lib.ncclAllGather(
                    ctypes.c_void_p(inputs[i].data_ptr()), count, data_type,
                    ctypes.c_void_p(outputs[i].data_ptr()), comm[i],
                    cudaStream()))


def reduce_scatter(inputs, outputs, op=SUM):
    _check_inputs(inputs, outputs, 1.0 / len(inputs))
    comm = communicator(inputs, outputs)
    count = inputs[0].numel() // len(inputs)
    data_type = nccl_types[inputs[0].type()]
    with torch.cuda._free_mutex():
        for i in range(len(inputs)):
            with torch.cuda.device(comm.devices[i]):
                check_error(lib.ncclReduceScatter(
                    ctypes.c_void_p(inputs[i].data_ptr()),
                    ctypes.c_void_p(outputs[i].data_ptr()), count, data_type,
                    op, comm[i], cudaStream()))


def _check_inputs(inputs, outputs=None, size_multiplier=1):
    devices = set()
    size = inputs[0].numel()
    if len(inputs) != len(outputs):
        raise ValueError('inputs and outputs must be the same length')
    for input, output in zip(inputs, outputs):
        if not input.is_cuda:
            raise TypeError('inputs must be CUDA inputs')
        if not input.is_contiguous():
            raise ValueError('inputs must be contiguous')
        device = input.get_device()
        if device in devices:
            raise ValueError('inputs must be on unique devices')
        devices.add(device)
        if input.numel() != size:
            raise ValueError('inputs must be the same size')

        if not output.is_contiguous():
            raise ValueError('outputs must be contiguous')
        if output.get_device() != device:
            raise ValueError('inputs and outputs must be on the same devices')
        if output.numel() != size * size_multiplier:
            raise ValueError(('incorrect output size; expected {0} but got {1}'
                              .format(size * size_multiplier, output.numel())))
