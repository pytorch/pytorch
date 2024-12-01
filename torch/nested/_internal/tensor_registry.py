import torch

from torch.utils.weak import WeakTensorKeyDictionary


class TensorRegistry:
    def __init__(self):
        self._tensor_to_id = WeakTensorKeyDictionary()
        self._next_id = 0

    def register(self, t, t_id=None):
        if t_id is None:
            t_id = self._next_id
            self._next_id += 1
        self._tensor_to_id[t] = t_id
        return t_id

    def try_get_int(self, tensor: torch.Tensor):
        return self._tensor_to_id.get(tensor)

    def copy(self):
        ret = TensorRegistry()
        ret._tensor_to_id = self._tensor_to_id.copy()
        ret._next_id = self._next_id


_global_tensor_registry = TensorRegistry()


# Make sure dynamo doesn't try to trace through this
def register_tensor(t, t_id=None):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
        return t.register_nested_int_id(t_id)
    else:
        return _global_tensor_registry.register(t, t_id=t_id)


def try_get_int(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
        return t.try_get_nested_int_id()
    else:
        return _global_tensor_registry.try_get_int(t)
