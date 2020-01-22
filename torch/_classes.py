import types
import torch._C

class _Classes(types.ModuleType):
    def __init__(self):
        super(_Classes, self).__init__('torch.classes')
        self.loaded_libraries = set()

    def __getattr__(self, attr):
        proxy = torch._C._get_custom_class_python_wrapper(attr)
        if proxy is None:
            raise RuntimeError('Class {} not registered!'.format(attr))
        return proxy

    def load_library(self, path):
        torch.ops.load_library(path)

# The classes "namespace"
classes = _Classes()
