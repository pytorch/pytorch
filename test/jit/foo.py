import torch
import jit.bar
# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script  # noqa: B903
class FooSameName(object):
    def __init__(self, x):
        self.x = x
        self.nested = jit.bar.FooSameName(x)
