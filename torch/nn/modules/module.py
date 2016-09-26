from collections import OrderedDict

import torch
from ..backends.thnn import backend as thnn_backend
from torch.autograd import Variable


class Module(object):

    def __init__(self, **parameters):
        self._backend = thnn_backend
        self._parameters = OrderedDict(parameters)
        self.backward_hooks = OrderedDict()
        self.forward_hooks = OrderedDict()
        self.train = True

    def forward(self, *input):
        raise NotImplementedError

    def type(self, type, *forwarded_args):
        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves,
                # and we don't want to create copy nodes.
                param._data = param.data.type(type, *forwarded_args)
        return self

    def cuda(self, device_id=None):
        import torch.cuda
        import torch.nn.cuda
        if device_id is not None:
            return self.type(torch.cuda.FloatTensor, device_id)
        else:
            return self.type(torch.cuda.FloatTensor)

    def float(self):
        return self.type(torch.FloatTensor)

    def double(self):
        return self.type(torch.DoubleTensor)

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
        return object.__getattribute__(self, name)

    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p not in memo:
                memo.add(p)
                yield p

    def zero_grad(self):
        for p in self.parameters():
            p.grad.zero_()

