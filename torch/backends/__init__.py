# mypy: allow-untyped-defs
import sys
import types
from contextlib import contextmanager

import torch


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


class _FP32Precision:
    def __init__(self, backend, op):
        self.backend = backend
        self.op = op

    def __setattr__(self, name, value):
        if name == "fp32_precision":
            torch._C._set_fp32_precision_setter(self.backend, self.op, value)
        elif name in ("backend", "op"):
            super().__setattr__(name, value)
        else:
            raise AttributeError("Unknown attribute " + name)

    def __getattr__(self, name):
        if name == "fp32_precision":
            return torch._C._get_fp32_precision_getter(self.backend, self.op)
        else:
            raise AttributeError("Unknown attribute " + name)


def set_flags(_fp32_precision="none"):
    orig_flags = (torch._C._get_fp32_precision_getter("generic", "all"),)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("generic", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


def _get_fp32_precision_getter(backend, op):
    def inner():
        return torch._C._get_fp32_precision_getter(backend, op)

    return inner


def _set_fp32_precision_setter(backend, op):
    def inner(precision):
        return torch._C._set_fp32_precision_setter(backend, op, precision)

    return inner


class GenericModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    fp32_precision = ContextProp(
        _get_fp32_precision_getter("generic", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)

from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    quantized as quantized,
)
