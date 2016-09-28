import torch.cuda
from .utils import THNN_H_PATH, THCUNN_H_PATH, parse_header, load_backend

class Backends(object):
    def __init__(self):
        self.backends = {}

    def __getattr__(self, name):
        return self.backends[name].load()

    def __getitem__(self, name):
        return self.backends[name].load()


class Backend(object):
    def __init__(self, t, lib, load_header, mixins=tuple()):
        self.t = t
        self.lib = lib
        self.load_header = load_header
        self.mixins = mixins
        self.backend = None

    def load(self):
        if self.backend is None:
            functions = self.load_header()
            self.backend = load_backend(self.t, self.lib, functions, self.mixins)
        return self.backend


class THNNCudaBackendStateMixin(object):
    @property
    def library_state(self):
        return torch.cuda._state_cdata


type2backend = Backends()

_thnn_headers = None
_thcunn_headers = None

def thnn_headers():
    global _thnn_headers
    if _thnn_headers is None:
        _thnn_headers = parse_header(THNN_H_PATH)
    return _thnn_headers

def thcunn_headers():
    global _thcunn_headers
    if _thcunn_headers is None:
        _thcunn_headers = parse_header(THCUNN_H_PATH)
    for function in _thcunn_headers:
        function.name = function.name[4:]
    return _thcunn_headers

for t in ['Float', 'Double']:
    backend = Backend(t, 'torch._thnn._THNN', thnn_headers)

    type2backend.backends['THNN{}Backend'.format(t)] = backend
    type2backend.backends['torch.{}Tensor'.format(t)] = backend
    type2backend.backends[getattr(torch, '{}Tensor'.format(t))] = backend

backend = Backend('Cuda', 'torch._thnn._THCUNN', thcunn_headers, (THNNCudaBackendStateMixin,))
type2backend.backends['THNNCudaBackend'] = backend
type2backend.backends['torch.cuda.FloatTensor'] = backend
type2backend.backends[torch.cuda.FloatTensor] = backend
