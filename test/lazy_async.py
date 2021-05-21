import torch
import torch.fx

# First a demonstration

class FakeAsyncType:
    """
    Just pretend this is an actual async type that references
    the potentially-not-yet-available value.
    """
    def __init__(self, actual_value):
        self.actual_value = actual_value

    def wait(self):
        # Pretend that we actually wait for a remote value here
        return self.actual_value


class GiveMeAsyncTensor(torch.nn.Module):
    def forward(self, x):
        return FakeAsyncType(torch.relu(x))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.give_async = GiveMeAsyncTensor()

    def forward(self, x):
        return torch.add(self.give_async(x), 3.0)


m = Model()
# m(torch.rand(3, 4))
"""
TypeError: add(): argument 'input' (position 1) must be Tensor, not FakeAsyncType
"""

# Now do it with a `__torch_function__` implementation on the async type

class FakeAsyncTypeTF:
    """
    This Async type has a `__torch_function__` implementation. What this means
    is that when this type is seen as an argument to a PyTorch function in a
    position where it expects a Tensor, the dispatcher will call into
    `FakeAsyncTypeTF.__torch_function__` for special handling.
    """
    def __init__(self, actual_value):
        self.actual_value = actual_value

    def wait(self):
        # Pretend that we actually wait for a remote value here
        return self.actual_value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        Our `__torch_function__` implementation goes through all of the
        args and kwargs and checks if any of them are `FakeAsyncTypeTF`.
        If it is, it will call `wait()` on it and replace the Async
        type object with the result of wait. In this way, async values
        are waited on when the concrete value is first needed and without
        the user having to write an explicit `wait()` call.
        """
        kwargs = kwargs or {}

        def wait_async(a):
            if isinstance(a, FakeAsyncTypeTF):
                return a.wait()
            else:
                return a
        # wait() on all FakeAsyncTypeTF args/kwargs and replace
        # them with the resulting value.
        new_args = torch.fx.node.map_aggregate(args, wait_async)
        new_kwargs = torch.fx.node.map_aggregate(kwargs, wait_async)

        return func(*new_args, *new_kwargs)

class GiveMeAsyncTensorTF(torch.nn.Module):
    def forward(self, x):
        return FakeAsyncTypeTF(torch.relu(x))

class ModelTF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.give_async = GiveMeAsyncTensorTF()

    def forward(self, x):
        return torch.add(self.give_async(x), 3.0)


m = ModelTF()
m(torch.rand(3, 4))
