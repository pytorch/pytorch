import threading
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

    def __init__(self, lib_prefix, lib_name, functions, mixins=tuple()):
        self.lib_prefix = lib_prefix
        self.lib_name = lib_name
        self.functions = functions
        self.mixins = mixins
        self.backend = None
        self.loading_lock = threading.Lock()

    def load(self):
        # This looks a little weird, but it's necessary for thread safe loading.
        # Loading the backend can take some time, so multiple threads can enter
        # the if clause. We have to ensure that only the first one to acquire
        # the lock will actually load the backend, and that the rest won't
        # do it again.
        if self.backend is None:
            with self.loading_lock:
                if self.backend is None:
                    lib = getattr(torch._C, self.lib_name)
                    self.backend = load_backend(self.lib_prefix, lib,
                                                self.functions, self.mixins)
        return self.backend


class THNNCudaBackendStateMixin(object):

    @property
    def library_state(self):
        return torch.cuda._state_cdata


type2backend = Backends()

_thnn_headers = parse_header(THNN_H_PATH)
_thcunn_headers = parse_header(THCUNN_H_PATH)

for t in ['Float', 'Double']:
    backend = Backend(t, '_THNN', _thnn_headers)

    type2backend.backends['THNN{}Backend'.format(t)] = backend
    type2backend.backends['torch.{}Tensor'.format(t)] = backend
    type2backend.backends[getattr(torch, '{}Tensor'.format(t))] = backend


for t in ['Half', '', 'Double']:
    backend = Backend('Cuda' + t, '_THCUNN', _thcunn_headers, (THNNCudaBackendStateMixin,))
    type2backend.backends['THNNCuda{}Backend'.format(t)] = backend
    py_name = 'Float' if t == '' else t
    type2backend.backends['torch.cuda.{}Tensor'.format(py_name)] = backend
    type2backend.backends[getattr(torch.cuda, '{}Tensor'.format(py_name))] = backend
