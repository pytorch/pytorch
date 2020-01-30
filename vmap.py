import torch
import functools
from torch import Tensor


HANDLED_FUNCTIONS_BATCHED = {}
VMAP_LAYER = 0

def implements_batched(torch_function):
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_BATCHED[torch_function] = func
        return func
    return decorator


class Batched(object):
    handled_functions = HANDLED_FUNCTIONS_BATCHED

    def __init__(self, value, bdim, layer=None):
        # Keep track of the value (tensor), batch dim, and which layer of nesting.
        self.value = value
        self.bdim = bdim
        if layer is None:
            self.layer = VMAP_LAYER
        else:
            self.layer = layer

    def __repr__(self):
        return "Batched_{}_{}_{}".format(self.value.size(), self.bdim, self.layer)

    def __torch_function__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in self.handled_functions:
            return NotImplemented
        return self.handled_functions[func](*args, **kwargs)

    def size(self):
        return self.value.size()

    def unsqueeze(self, dim):
        return Batched(
            self.value.unsqueeze(dim),
            self.bdim if self.bdim < dim else self.bdim + dim + 1,
            self.layer)


def lift_to_layer(arr, layer):
    if isinstance(arr, Batched) and arr.layer == layer:
        return arr
    else:
        return Batched(arr, None)


@implements_batched(torch.add)
def add(arr1, arr2):
    max_layer = max([arr.layer for arr in [arr1, arr2] if hasattr(arr, 'layer')])
    arr1 = lift_to_layer(arr1, max_layer)
    arr2 = lift_to_layer(arr2, max_layer)

    # Unwrap one vmap nesting layer.
    result = add_batched([arr1.value, arr2.value], [arr1.bdim, arr2.bdim])
    return Batched(*result, layer=max_layer)


def move_bdim_to_front(x, bdim):
    if bdim is None:
        return x.unsqueeze(0)
    return x.transpose(bdim, 0)


def add_batched(args, dims):
    inputs = [move_bdim_to_front(arg, bdim) for arg, bdim in zip(args, dims)]
    output = torch.add(*inputs)
    return (output, 0)


def vmap(fn, in_axes):
    global VMAP_LAYER
    VMAP_LAYER += 1

    def wrapped(*args):
        global VMAP_LAYER
        batched_inputs = [Batched(arg, dim) for arg, dim in zip(args, in_axes)]
        output = fn(*batched_inputs)
        VMAP_LAYER -= 1
        return output.value
    return wrapped


x3 = torch.ones(3)
x23 = torch.ones(2, 3)
x53 = torch.ones(5, 3)

# Elementwise
output = vmap(torch.add, [0, 0])(x23, x23)
assert list(output.shape) == [2, 3]

# Batched + unbatched
output = vmap(torch.add, [0, None])(x23, x3)
assert list(output.shape) == [2, 3]

# Nested
output = vmap(lambda xx: vmap(lambda yy: torch.add(xx, yy), [0])(x53), [0])(x23)
assert list(output.shape) == [2, 5, 3]
