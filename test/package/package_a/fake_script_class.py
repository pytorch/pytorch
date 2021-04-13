import torch

@torch.jit.script
class MyScriptClass:  # flake8: noqa
    """Intended to be scripted."""
    def __init__(self, x):
        self.foo = x

    def set_foo(self, x):
        self.foo = x

@torch.jit.script
def uses_script_class(x):
    """Intended to be scripted."""
    foo = MyScriptClass(x)
    return foo.foo
