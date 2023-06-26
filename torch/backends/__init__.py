from contextlib import contextmanager
import types
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
            raise RuntimeError("not allowed to set %s flags "
                               "after disable_global_flags; please use flags() context manager instead" % obj.__name__)

class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


from torch.backends import cpu as cpu
from torch.backends import cuda as cuda
from torch.backends import mps as mps
from torch.backends import cudnn as cudnn
from torch.backends import mkl as mkl
from torch.backends import mkldnn as mkldnn
from torch.backends import openmp as openmp
from torch.backends import quantized as quantized
