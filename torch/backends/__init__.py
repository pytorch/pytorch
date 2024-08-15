# mypy: allow-untyped-defs
import types
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
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
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
    cusparselt as cusparselt,
    mha as mha,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    quantized as quantized,
    xeon as xeon,
)
