import types
import torch._C

class _Classes(types.ModuleType):
    def __init__(self):
        super(_Classes, self).__init__('torch.classes')

    def __getattr__(self, attr):
        proxy = torch._C._get_custom_class_python_wrapper(attr)
        setattr(proxy, '__torchscript_custom_class_qualname', 'torch.classes.' + attr)
        if proxy is None:
            raise RuntimeError('Class {} not registered!'.format(attr))
        return proxy

# The classes "namespace"
classes = _Classes()
