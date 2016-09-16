import torch.cuda
import torch._thnn._THCUNN
from .utils import THCUNN_H_PATH, parse_header, load_backend
from . import type2backend

class THNNCudaBackendStateMixin(object):
    @property
    def library_state(self):
        return torch.cuda._state_cdata

generic_functions = parse_header(THCUNN_H_PATH)
# Type will be appended in load_backend
for function in generic_functions:
    function.name = function.name[4:]

backend = load_backend('Cuda', torch._thnn._THCUNN, generic_functions, 'torch._thnn.thcunn', (THNNCudaBackendStateMixin,))
type2backend['torch.cuda.FloatTensor'] = backend
type2backend[torch.cuda.FloatTensor] = backend
