import torch
from torch._C import _ImperativeEngine as ImperativeEngine


__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()


compiled_autograd_final_callbacks = []


def queue_callback(cb):
    global compiled_autograd_final_callbacks
    compiled_autograd_final_callbacks.append(cb)


def exec_post_processing():
    # TODO(yf225): use lock to be thread-safe (and local to the graph)
    global compiled_autograd_final_callbacks
    for cb in compiled_autograd_final_callbacks:
        cb()
    compiled_autograd_final_callbacks.clear()