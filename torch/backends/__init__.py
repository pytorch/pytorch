import sys
import types
import torch
from contextlib import contextmanager

# The idea for this parameter is that we forbid bare assignment
# to torch.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                "not allowed to set %s flags "
                "after disable_global_flags; please use flags() context manager instead"
                % obj.__name__
            )


class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    openmp as openmp,
    quantized as quantized,
)

def _register_backend_module(device_type, module):
    r"""Register an external runtime module of the specific :attr:`device_type`
    supported by torch.

    After the :attr:`module` is registered correctly, the user can refer
    the external runtime module as part of torch with attribute torch.backends.xxx.
    """
    # Make sure the device_type represent a supported device type for torch.
    device_type = torch.device(device_type).type
    m = sys.modules[__name__]
    if hasattr(m, device_type):
        raise RuntimeError("The runtime module of '{}' has already "
                           "been registered with '{}'".format(device_type, getattr(m, device_type)))
    setattr(m, device_type, module)
    torch_module_name = '.'.join([__name__, device_type])
    sys.modules[torch_module_name] = module
