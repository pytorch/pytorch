import ctypes
import torch.cuda
from torch.legacy.nn.ffi import parse_header, load_backend, type2backend, _backends

THCUNN_H_PATH = '/home/apaszke/pytorch_cuda/torch/legacy/cunn/THCUNN.h'
THCUNN_LIB_PATH = '/home/apaszke/torch/install/lib/lua/5.1/libTHCUNN.so'

class THNNCudaBackendStateMixin(object):
    @property
    def library_state(self):
        return ctypes.c_void_p(torch.cuda._state_cdata)

generic_functions = parse_header(THCUNN_H_PATH)
# Type will be appended in load_backend
for function in generic_functions:
    function.name = function.name[4:]

lib_handle = ctypes.cdll.LoadLibrary(THCUNN_LIB_PATH)
load_backend('Cuda', lib_handle, generic_functions, (THNNCudaBackendStateMixin,))
type2backend['torch.cuda.FloatTensor'] = _backends.THNNCudaBackend
