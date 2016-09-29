from itertools import chain
from collections import OrderedDict

import torch
from ..backends.thnn import backend as thnn_backend
from torch.autograd import Variable


class Module(object):

    def __init__(self, **parameters):
        self._backend = thnn_backend
        self._parameters = OrderedDict(parameters)
        self._buffers = {}
        self.backward_hooks = OrderedDict()
        self.forward_hooks = OrderedDict()
        self.train = True

    def forward(self, *input):
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def _apply(self, fn):
        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param._data = fn(param.data)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def cuda(self, device_id=None):
        return self._apply(lambda t: t.cuda(device_id))

    def cpu(self, device_id=None):
        return self._apply(lambda t: t.cpu())

    def float(self):
        return self._apply(lambda t: t.float())

    def double(self):
        return self._apply(lambda t: t.double())

    def register_backward_hook(self, name, hook):
        assert name not in self.backward_hooks, \
            "Trying to register a second backward hook with name {}".format(name)
        self.backward_hooks[name] = hook

    def remove_backward_hook(self, name):
        assert name in self.backward_hooks, \
            "Trying to remove an inexistent backward hook with name {}".format(name)
        del self.backward_hooks[name]

    def register_forward_hook(self, name, hook):
        assert name not in self.forward_hooks, \
            "Trying to register a second forward hook with name {}".format(name)
        self.forward_hooks[name] = hook

    def remove_forward_hook(self, name):
        assert name in self.forward_hooks, \
            "Trying to remove an inexistent forward hook with name {}".format(name)
        del self.forward_hooks[name]

    def __call__(self, *input):
        result = self.forward(*input)
        for hook in self.forward_hooks.values():
            hook(self, input, result)
        if isinstance(result, tuple):
            fn = result[0].creator
        else:
            fn = result.creator
        for key, hook in self.backward_hooks.items():
            fn.register_hook(key, lambda gi,go,hook=hook: hook(self, gi, go))
        return result

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                if not isinstance(value, Variable):
                    raise RuntimeError(("assiging a {} object as parameter {} "
                        "- you can only assign Variables as model"
                        "parameters").format(type(value), name))
                if value.creator:
                    raise RuntimeError(("All parameters should be leaf "
                            "variables - they should be created explicitly, "
                            "not as a result of computation on other "
                            "variables. If you want to express {} as a "
                            "function of another variable, simply repeat the "
                            "computation at every forward pass.").format(name))
        return object.__setattr__(self, name, value)

    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                yield p

    def zero_grad(self):
        for p in self.parameters():
            p.grad.zero_()

