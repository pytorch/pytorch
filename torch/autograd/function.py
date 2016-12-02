import torch
import torch._C as _C
from collections import OrderedDict
from itertools import chain


class Function(_C._FunctionBase):

    __call__ = _C._FunctionBase._do_forward

    def save_for_backward(self, *tensors):
        self.to_save = tensors

    def mark_dirty(self, *args):
        self.dirty_tensors = args

    def mark_shared_storage(self, *pairs):
        self.shared_pairs = pairs

    def mark_non_differentiable(self, *args):
        self.non_differentiable = args

    def register_hook(self, name, hook):
        self._backward_hooks = self._backward_hooks or OrderedDict()
        assert name not in self._backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        self._backward_hooks[name] = hook

    def remove_hook(self, name):
        assert self._backward_hooks and name in self._backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self._backward_hooks[name]

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace

def _nested_map(condition, fn):
    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_map(x) for x in obj)
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                "an input object of type " + torch.typename(obj))
    return _map

def _iter_filter(condition):
    def _iter(obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                "an input object of type " + torch.typename(obj))
    return _iter


_iter_variables = _iter_filter(lambda o: isinstance(o, torch.autograd.Variable))
_iter_tensors = _iter_filter(torch.is_tensor)
_iter_None_tensors = _iter_filter(lambda o: o is None or torch.is_tensor(o))
_map_variable_tensor = _nested_map(lambda o: isinstance(o, torch.autograd.Variable), lambda o: o.data)

def _map_tensor_fromiter(itr):
     return _nested_map(lambda o: torch.is_tensor(o), lambda o: next(itr))

class NestedIOFunction(Function):

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_variables(input))
        flat_output = super(NestedIOFunction, self)._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_variables = _map_tensor_fromiter(iter(flat_output))(self._nested_output)
        return nested_variables

    def backward(self, *gradients):
        nested_gradients = _map_tensor_fromiter(iter(gradients))(self._nested_output)
        del self._nested_output
        result = self.backward_extended(*nested_gradients)
        del self._to_save_nested
        return tuple(_iter_None_tensors(result))

    __call__ = _do_forward

    def forward(self, *args):
        nested_tensors = _map_variable_tensor(self._nested_input)
        result = self.forward_extended(*nested_tensors)
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))

    def save_for_backward(self, *args):
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    @property
    def saved_tensors(self):
        flat_tensors = super(NestedIOFunction, self).saved_tensors
        return _map_tensor_fromiter(iter(flat_tensors))(self._to_save_nested)

    def mark_dirty(self, *args, **kwargs):
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args, **kwargs):
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    def forward_extended(self, *input):
        raise NotImplementedError

    def backward_extended(self, *grad_output):
        raise NotImplementedError
