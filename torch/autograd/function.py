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
        self.backward_hooks = self.backward_hooks or OrderedDict()
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        self.backward_hooks[name] = hook

    def remove_hook(self, name):
        assert self.backward_hooks and name in self.backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self.backward_hooks[name]

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace

