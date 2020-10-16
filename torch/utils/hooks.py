from collections import OrderedDict
import weakref
import warnings
from typing import Any


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return (self.hooks_dict_ref(), self.id)

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self) -> 'RemovableHandle':
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def unserializable_hook(f):
    """
    Decorator which marks a function as an unserializable hook.
    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
    f.__torch_unserializable__ = True
    return f


def warn_if_has_hooks(tensor):
    if tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(k, "__torch_unserializable__"):
                warnings.warn("backward hook {} on tensor will not be "
                              "serialized.  If this is expected, you can "
                              "decorate the function with @torch.utils.hooks.unserializable_hook "
                              "to suppress this warning".format(repr(hook)))
