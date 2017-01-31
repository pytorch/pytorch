import torch._C as _C
import ctypes
import warnings
import torch.cuda
import sys
import os.path as path

enabled = True  # set to False to globally disable cuDNN

lib = None
# TODO: fix libname for Windows
# TODO: dynamic version checks via cudnnGetVersion
# TODO: load 5.1.3 if using CUDA 7.5 and 5.1.5 if using CUDA 8.0
thisdir = path.dirname(__file__)
libpaths = ['', path.join(thisdir, '../../lib')]
if sys.platform.startswith('linux'):
    libnames = ['libcudnn.so.6.0.5', 'libcudnn.so.6.0.10', 'libcudnn.so.5.1.5', 'libcudnn.so.5.1.3',
                'libcudnn.so.5.0.5', 'libcudnn.so.5.1.10']
elif sys.platform == 'darwin':
    libnames = ['libcudnn.6.dylib', 'libcudnn.5.dylib']
else:
    libnames = []


def _loadlib():
    global lib
    loaded = False
    for libpath in libpaths:
        for libname in libnames:
            try:
                lib = ctypes.cdll.LoadLibrary(path.join(libpath, libname))
                loaded = True
                break
            except OSError:
                continue
        if loaded:
            break
    if loaded:
        lib.cudnnGetErrorString.restype = ctypes.c_char_p
    else:
        lib = None
        raise OSError("Could not load cuDNN")


def is_acceptable(tensor):
    if not enabled:
        return False
    if not (isinstance(tensor, torch.cuda.HalfTensor) or
            isinstance(tensor, torch.cuda.FloatTensor) or
            isinstance(tensor, torch.cuda.DoubleTensor)):
        return False
    if lib is None:
        try:
            _loadlib()
        except Exception:
            warnings.warn('cuDNN library not found. Check your {libpath}'.format(
                libpath={
                    'darwin': 'DYLD_LIBRARY_PATH',
                    'win32': 'PATH'
                }.get(sys.platform, 'LD_LIBRARY_PATH')))
            return False
    if not _C.has_cudnn:
        warnings.warn("cuDNN library has been detected, but your pytorch "
                      "installation was compiled without support for it. You "
                      "might want to rebuild pytorch, making sure the library "
                      "is visible to the build system.")
        return False
    return True

__cudnn_version = []


def version():
    if not lib:
        raise RuntimeError("cuDNN not initialized")
    if len(__cudnn_version) == 0:
        __cudnn_version.append(lib.cudnnGetVersion())
    return __cudnn_version[0]

_handles = {}

benchmark = False
verbose = False
workspace_limit = None

CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1
CUDNN_DATA_HALF = 2

CUDNN_CONVOLUTION = 0
CUDNN_CROSS_CORRELATION = 1

CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2

CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1

CUDNN_RNN_RELU = 0
CUDNN_RNN_TANH = 1
CUDNN_LSTM = 2
CUDNN_GRU = 3

CUDNN_LINEAR_INPUT = 0
CUDNN_SKIP_INPUT = 1

CUDNN_NON_DETERMINISTIC = 0
CUDNN_DETERMINISTIC = 1

CUDNN_RNN_ALGO_STANDARD = 0
CUDNN_RNN_ALGO_PERSIST_STATIC = 1
CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2


class CuDNNHandle:

    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreate(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroy(self))


class CuDNNError(RuntimeError):

    def __init__(self, status):
        self.status = status
        msg = '{}: {}'.format(status, get_error_string(status))
        super(CuDNNError, self).__init__(msg)


class TensorDescriptor(object):

    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateTensorDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroyTensorDescriptor(self._as_parameter_))
        del self._as_parameter_

    def set(self, tensor):
        self._type = tensor.type()
        self._size = tensor.size()
        self._stride = tensor.stride()
        check_error(lib.cudnnSetTensorNdDescriptor(
            self, _typemap[tensor.type()], tensor.dim(),
            int_array(tensor.size()), int_array(tensor.stride())))

    def as_tuple(self):
        return (self._type, tuple(self._size), tuple(self._stride))


class TensorDescriptorArray(object):

    def __init__(self, N):
        self.ptrs = (ctypes.c_void_p * N)()
        for i in range(N):
            ptr = ctypes.byref(self.ptrs, i * ctypes.sizeof(ctypes.c_void_p))
            check_error(lib.cudnnCreateTensorDescriptor(ptr))
        self._as_parameter_ = self.ptrs

    def __del__(self):
        for ptr in self.ptrs:
            check_error(lib.cudnnDestroyTensorDescriptor(ctypes.c_void_p(ptr)))

    def __getitem__(self, key):
        return ctypes.c_void_p(self.ptrs[key])

    def set(self, tensor):
        self._type = tensor.type()
        self._size = tensor.size()
        self._stride = tensor.stride()
        for ptr in self.ptrs:
            check_error(lib.cudnnSetTensorNdDescriptor(
                ctypes.c_void_p(ptr), _typemap[tensor.type()], tensor.dim(),
                int_array(tensor.size()), int_array(tensor.stride())))

    def as_tuple(self):
        return (self._type, tuple(self._size), tuple(self._stride))


class ConvolutionDescriptor(object):

    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateConvolutionDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroyConvolutionDescriptor(self._as_parameter_))
        del self._as_parameter_

    def set(self, typename, pad, stride):
        self._pad = pad
        self._stride = stride
        upscale = int_array([1, 1])
        check_error(lib.cudnnSetConvolutionNdDescriptor(
            self, 2, int_array(pad), int_array(stride), upscale,
            CUDNN_CROSS_CORRELATION, _typemap[typename]))

    def as_tuple(self):
        return (self._pad, self._stride)


class FilterDescriptor(object):

    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateFilterDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroyFilterDescriptor(self._as_parameter_))
        del self._as_parameter_

    def set(self, weight):
        self._size = weight.size()
        datatype = _typemap[weight.type()]
        check_error(lib.cudnnSetFilterNdDescriptor(
            self, datatype, CUDNN_TENSOR_NCHW, weight.ndimension(), int_array(weight.size())))

    def as_tuple(self):
        return tuple(self._size)


class DropoutDescriptor(object):

    def __init__(self, handle, dropout, seed):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateDropoutDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

        dropout_states_size = ctypes.c_long()
        check_error(lib.cudnnDropoutGetStatesSize(
            handle,
            ctypes.byref(dropout_states_size)))

        self.state = torch.cuda.ByteTensor(dropout_states_size.value)

        check_error(lib.cudnnSetDropoutDescriptor(
            self,
            handle,
            ctypes.c_float(dropout),
            ctypes.c_void_p(self.state.data_ptr()),
            ctypes.c_size_t(self.state.size(0)),
            ctypes.c_ulonglong(seed),
        ))

    def __del__(self):
        check_error(lib.cudnnDestroyDropoutDescriptor(self))


class RNNDescriptor(object):

    def __init__(self, handle, hidden_size, num_layers, dropout_desc, input_mode,
                 bidirectional, mode, datatype):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateRNNDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr
        if version() >= 6000:
            check_error(lib.cudnnSetRNNDescriptor_v6(
                handle,
                self,
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                bidirectional,
                mode,
                CUDNN_RNN_ALGO_STANDARD,
                datatype
            ))
        else:
            check_error(lib.cudnnSetRNNDescriptor(
                self,
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                bidirectional,
                mode,
                datatype
            ))

    def __del__(self):
        check_error(lib.cudnnDestroyRNNDescriptor(self))


class ConvolutionAlgoPerf_v5(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
    ]


class ConvolutionAlgoPerf_v6(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_int),
        ("status", ctypes.c_int),
        ("time", ctypes.c_float),
        ("memory", ctypes.c_size_t),
        ("determinism", ctypes.c_int),
        ("reserved", ctypes.c_int * 4)
    ]


def check_error(status):
    if status is not 0:
        raise CuDNNError(status)


def get_error_string(status):
    return lib.cudnnGetErrorString(status)


def get_handle():
    if lib is None:
        _loadlib()
    current_device = torch.cuda.current_device()
    handle = _handles.get(current_device, None)
    if handle is None:
        handle = CuDNNHandle()
        _handles[current_device] = handle
    return handle

_typemap = {
    'torch.cuda.HalfTensor': CUDNN_DATA_HALF,
    'torch.cuda.FloatTensor': CUDNN_DATA_FLOAT,
    'torch.cuda.DoubleTensor': CUDNN_DATA_DOUBLE,
}

_sizeofmap = {
    CUDNN_DATA_HALF: 2,
    CUDNN_DATA_FLOAT: 4,
    CUDNN_DATA_DOUBLE: 8,
}


def c_type(tensor):
    if isinstance(tensor, torch.cuda.HalfTensor):
        return ctypes.c_float
    elif isinstance(tensor, torch.cuda.FloatTensor):
        return ctypes.c_float
    elif isinstance(tensor, torch.cuda.DoubleTensor):
        return ctypes.c_double
    else:
        raise ValueError("unknown type '{}'".format(type(tensor)))


def int_array(itr):
    array_type = ctypes.c_int * len(itr)
    return array_type(*itr)


def descriptor(tensor, N=None):
    if N is not None:
        descriptor = TensorDescriptorArray(N)
    else:
        descriptor = TensorDescriptor()
    if tensor.dim() == 2:
        tensor = tensor.view(tensor.size(0), tensor.size(1), 1, 1)
    elif tensor.dim() == 3:
        tensor = tensor.view(tensor.size(0), tensor.size(1), tensor.size(2), 1)
    descriptor.set(tensor)
    return descriptor

_autotuner_forward = {}
_autotuner_backward_data = {}
_autotuner_backward_filter = {}


def convolution_autotuner_key(idesc, weight_desc, conv_desc):
    return (idesc.as_tuple(), weight_desc.as_tuple(), conv_desc.as_tuple())


def convolution_forward_algorithm(idesc, weight_desc, conv_desc, odesc):
    k = convolution_autotuner_key(idesc, weight_desc, conv_desc)
    if k in _autotuner_forward:
        return _autotuner_forward[k]

    if benchmark:
        if version() < 6000:
            perf_results = ConvolutionAlgoPerf_v5()
        else:
            perf_results = ConvolutionAlgoPerf_v6()
        algo_count = ctypes.c_int()
        check_error(lib.cudnnFindConvolutionForwardAlgorithm(
            get_handle(), idesc, weight_desc, conv_desc, odesc, 1,
            ctypes.byref(algo_count), ctypes.byref(perf_results)))
        _autotuner_forward[k] = perf_results.algo
        return perf_results.algo

    search_mode = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
    wlimit = 0
    if workspace_limit is not None:
        wlimit = workspace_limit
        search_mode = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT

    fwd_alg = ctypes.c_int()
    check_error(lib.cudnnGetConvolutionForwardAlgorithm(
        get_handle(), idesc, weight_desc, conv_desc, odesc, search_mode,
        wlimit, ctypes.byref(fwd_alg)))
    return fwd_alg


def convolution_forward_workspace_size(*args):
    check_error(lib.cudnnGetConvolutionForwardWorkspaceSize(*args))


def convolution_forward(*args):
    check_error(lib.cudnnConvolutionForward(*args))


def convolution_backward_data(*args):
    return check_error(lib.cudnnConvolutionBackwardData(*args))


def convolution_backward_data_algorithm(weight_desc, odesc, conv_desc, idesc):
    k = convolution_autotuner_key(idesc, weight_desc, conv_desc)
    if k in _autotuner_backward_data:
        return _autotuner_backward_data[k]

    if benchmark:
        perf_results = ConvolutionAlgoPerf()
        algo_count = ctypes.c_int()
        check_error(lib.cudnnFindConvolutionBackwardDataAlgorithm(
            get_handle(), weight_desc, odesc, conv_desc, idesc, 1,
            ctypes.byref(algo_count), ctypes.byref(perf_results)))
        _autotuner_backward_data[k] = perf_results.algo
        return perf_results.algo

    search_mode = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
    wlimit = 0
    if workspace_limit is not None:
        wlimit = workspace_limit
        search_mode = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

    bwd_data_alg = ctypes.c_int()
    check_error(lib.cudnnGetConvolutionBackwardDataAlgorithm(
        get_handle(), weight_desc, odesc, conv_desc, idesc, search_mode,
        wlimit, ctypes.byref(bwd_data_alg)))
    return bwd_data_alg


def convolution_backward_data_workspace_size(*args):
    return check_error(lib.cudnnGetConvolutionBackwardDataWorkspaceSize(*args))


def convolution_backward_filter(*args):
    return check_error(lib.cudnnConvolutionBackwardFilter(*args))


def convolution_backward_filter_algorithm(idesc, odesc, conv_desc, weight_desc):
    k = convolution_autotuner_key(idesc, weight_desc, conv_desc)
    if k in _autotuner_backward_filter:
        return _autotuner_backward_filter[k]

    if benchmark:
        perf_results = ConvolutionAlgoPerf()
        algo_count = ctypes.c_int()
        check_error(lib.cudnnFindConvolutionBackwardFilterAlgorithm(
            get_handle(), idesc, odesc, conv_desc, weight_desc, 1,
            ctypes.byref(algo_count), ctypes.byref(perf_results)))
        _autotuner_backward_filter[k] = perf_results.algo
        return perf_results.algo

    search_mode = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
    wlimit = 0
    if workspace_limit is not None:
        wlimit = workspace_limit
        search_mode = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT

    bwd_filter_alg = ctypes.c_int()
    check_error(lib.cudnnGetConvolutionBackwardFilterAlgorithm(
        get_handle(), idesc, odesc, conv_desc, weight_desc, search_mode,
        wlimit, ctypes.byref(bwd_filter_alg)))
    return bwd_filter_alg


def convolution_backward_filter_workspace_size(*args):
    return check_error(lib.cudnnGetConvolutionBackwardFilterWorkspaceSize(*args))


def convolution_backward_bias(*args):
    check_error(lib.cudnnConvolutionBackwardBias(*args))


def add_tensor(*args):
    check_error(lib.cudnnAddTensor(*args))
