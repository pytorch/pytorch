from typing import Optional, Sequence, Tuple
from numbers import Number
import collections
import math
import inspect
from itertools import chain
import functools
import types
import string

import torch
from torch.fx.graph import Graph

class ProxyContext(object):
    def __init__(self):
        self.graph = Graph()

        # Private attributes for generating names
        self._tensor_name_counter = 0
        self._dim_name_counter = 0
        self._shape_name_counter = 0
        self._lowercase = tuple(string.ascii_lowercase)
        self._uppercase = tuple(string.ascii_uppercase)

    @staticmethod
    def _create_name(idx, chars):
        name = ""
        while idx > len(chars):
            name = chars[idx % len(chars)] + name
            idx = idx - len(chars)
        name = chars[idx] + name

        return name

    # TODO: different types of names
    def tensor_name(self):
        idx = self._tensor_name_counter
        self._tensor_name_counter = self._tensor_name_counter + 1

        return self._create_name(idx, self._lowercase)

    def dim_name(self):
        idx = self._dim_name_counter
        self._dim_name_counter = self._dim_name_counter + 1

        return "d" + self._create_name(idx, self._uppercase)

    def shape_name(self):
        idx = self._shape_name_counter
        self._shape_name_counter = self._shape_name_counter + 1

        return "s" + self._create_name(idx, self._uppercase)


# Proxies
class DimProxy(object):
    def __init__(self, ctx, length: Optional[Number]=None):
        self.ctx = ctx
        self.name = ctx.dim_name()
        self.length = length
        self.equivalent_to = None

    def __repr__(self):
        if self.length is not None:
            return str(self.length)
        elif self.equivalent_to is not None:
            return str(self.equivalent_to.name)
        else:
            return str(self.name)

    def _remap(self, equiv):
        assert self.name != equiv.name

        if self.equivalent_to is not None:
            self.equivalent_to._remap(equiv)

        self.equivalent_to = equiv

    def require_same_length(self, other):
        # Type checks
        assert isinstance(other, DimProxy)

        # Attempts direct comparison
        try:
            if self != other:
                raise RuntimeError("Attempting to require two dimensions of different lengths have the same length!")
        except ValueError:
            pass

        def _equiv(x):
            if x.equivalent_to is not None:
                return x.equivalent_to.name
            return x.name

        a, b = _equiv(self), _equiv(other)
        if a == b:
            pass
        elif a < b:
            other._remap(self.equivalent_to if self.equivalent_to is not None else self)
        else:
            # a > b
            self._remap(other.equivalent_to if other.equivalent_to is not None else other)

    @staticmethod
    def _len(x):
        assert isinstance(x, (Number, DimProxy))

        if isinstance(x, Number):
            return x

        # x is DimProxy
        return x.length

    # Comparison operators
    def __eq__(self, other):
        a, b = self.length, self._len(other)

        if a is not None and b is not None:
            return (a == b)
        if (a is not None and a == 1) or (b is not None and b == 1):
            return False
        if b is not None and b < 0:
            return False

        raise ValueError("Can't determine equality!")

    def __ne__(self, other):
        return not (self == other)

    def __ge__(self, other):
        a, b = self.length, self._len(other)

        if a is not None and b is not None:
            return (a >= b)
        if b is not None and b < 0:
            return True

        raise ValueError("Can't determine greater than or equal!")

    def __gt__(self, other):
        a, b = self.length, self._len(other)

        if a is not None and b is not None:
            return (a > b)
        if b is not None and b < 0:
            return True

        raise ValueError("Can't determine greater than!")

    def __le__(self, other):
        a, b = self.length, self._len(other)

        if a is not None and b is not None:
            return (a <= b)
        if b is not None and b < 0:
            return False

        raise ValueError("Can't determine less than or equal!")

    def __lt__(self, other):
        a, b = self.length, self._len(other)

        if a is not None and b is not None:
            return (a < b)
        if b is not None and b <= 0:
            return False

        raise ValueError("Can't determine less than!")

class ShapeProxy(collections.abc.Sequence):
    def __init__(self, ctx, node, name, shape):
        assert isinstance(shape, Sequence)
        self.ctx = ctx
        self.node = node
        self.name = name

        if isinstance(shape, ShapeProxy):
            self._shape = shape._shape
            return

        def _dim_helper(x):
            # Type and value checks
            assert isinstance(x, (Number, DimProxy))

            if isinstance(x, DimProxy):
                return x

            # x is a Number
            if x < 0:
                raise ValueError("Trying to construct shape with negative length dimension!")

            if x == 1:
                return DimProxy(ctx, 1)

            return DimProxy(ctx)

        self._shape = tuple(_dim_helper(x) for x in shape)

    def __repr__(self):
        return self.name

    # Sequence interface
    def __getitem__(self, index):
        return self._shape[index]

    def __len__(self):
        return len(self._shape)

    def count(self, value):
        return self._shape.count(value)

    def index(self, value):
        return self._shape.index(value)

    def __contains__(self, value):
        return value in self._shape

    def __iter__(self):
        return iter(self._shape)

    def __reversed__(self):
        return reversed(self._shape)


# Add base ptr, base ptr alignment, contiguity... ?
class TensorProxy(object):
    # TODO: allow direct construction or construction from an actual tensor
    def __init__(self, ctx, node, name, *, tensor=None, shape=None, dtype=None, device=None):
        assert isinstance(ctx, ProxyContext)
        self.ctx = ctx
        self.node = node
        self.name = name

        shape_name = self.name + ".shape"
        if tensor is not None:
            assert isinstance(tensor, (torch.Tensor, TensorProxy))
            self._shape = ShapeProxy(ctx, node, shape_name, tensor.shape) if shape is None else ShapeProxy(ctx, shape)
            self._dtype = tensor.dtype if dtype is None else dtype
            self._device = tensor.device if device is None else device
        else:
            self._shape = ShapeProxy(ctx, node, shape_name, shape)
            self._dtype = dtype
            self._device = device

    # TODO: extend
    def __repr__(self):
        return self.name

        # s = "TensorProxy{"

        # s += str(self.name) + ", "
        # s += str(self._shape) + ", "
        # s += str(self._dtype) + ", "
        # s += str(self._device)

        # return s + "}"

    # TODO: consider overriding in the future
    def __getattribute__(self, name):
        return super().__getattribute__(name)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

# TODO: NumberProxy, SequenceProxy, ...

# Types
TensorLikes = (torch.Tensor, TensorProxy)

#TODO: Make this return a functional that implements the trace caching scheme
def trace(op, *args):
    ctx = ProxyContext()

    for arg in args:
        # TODO: extend to support additional types and make this generic
        assert isinstance(arg, TensorLikes)

    def _maybe_proxy(x):
        if isinstance(x, TensorProxy):
            return x
        # x is torch.Tensor
        name = ctx.tensor_name()
        node = ctx.graph.placeholder(str(name))
        result = TensorProxy(ctx, node, name, tensor=x)
        return result

    proxy_args = tuple(_maybe_proxy(arg) for arg in args)

    result = op(*proxy_args)
    output_node = ctx.graph.output(result)
    result.node.users[output_node] = None

    ctx.graph.eliminate_dead_code()
    return ctx.graph