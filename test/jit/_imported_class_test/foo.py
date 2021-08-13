import torch
from . import bar
from .bar import f
# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script  # noqa: B903
class FooSameName(object):
    def __init__(self, x):
        self.x = x
        self.nested = bar.FooSameName(x)


class MyObj(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return f(x)


def simple_func(x):
    return f(x)


class Nada:
    pass


class NoConstructor:

    def g(self, x):
        return f(x)


def inner():

    def xyz(x):
        return x

    class MyObj(object):
        def __init__(self):
            pass

        def __call__(self, x):
            return xyz(x)

    return MyObj
