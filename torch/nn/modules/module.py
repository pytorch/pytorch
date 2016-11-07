from itertools import chain
from collections import OrderedDict

import torch
from ..backends.thnn import backend as thnn_backend
from ..parameter import Parameter
from torch.autograd import Variable


class Module(object):

    def __init__(self, **parameters):
        self._backend = thnn_backend
        self._parameters = OrderedDict(parameters)
        self._buffers = {}
        self.backward_hooks = OrderedDict()
        self.forward_hooks = OrderedDict()
        self.training = True
        for name, param in self._parameters.items():
            if not isinstance(param, Parameter):
                if isinstance(param, Variable):
                    raise TypeError("can't use a Variable as a module "
                        "parameter.  Convert it to torch.nn.Parameter first.")
                if param is not None:
                    param = Parameter(param)
            self._parameters[name] = param

    def forward(self, *input):
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def _apply(self, fn):
        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param.grad is not None:
                    param._grad = fn(param._grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def apply(self, fn):
        fn(self)
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
        var = result
        while not isinstance(var, Variable):
            var = var[0]
        creator = var.creator
        for key, hook in self.backward_hooks.items():
            creator.register_hook(key, lambda gi,go,hook=hook: hook(self, gi, go))
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
        _parameters = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if _parameters is None:
                raise AttributeError(
                    "cannot assign parameter before Module.__init__() call")
            if value.creator:
                raise ValueError(
                    "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                    "parameters must be created explicitly. To express '{0}' "
                    "as a function of another variable, compute the value in "
                    "the forward() method.".format(name))
            _parameters[name] = value
        elif _parameters and name in _parameters:
            if value is not None:
                raise TypeError("cannot assign '{}' object to parameter '{}' "
                                "(torch.nn.Parameter or None required)"
                                .format(torch.typename(value), name))
            _parameters[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        else:
            object.__delattr__(self, name)

    def parameter_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param
        return destination

    def load_parameter_dict(self, param_dict):
        for name, param in self._parameters.items():
            self._parameters[name] = param_dict.get(name, param)

    def parameters(self, memo=None):
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                yield p

    def children(self):
        if False:
            yield

    def modules(self, memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self

    def train(self):
        self.training = True
        for p in self._parameters.values():
            if p is not None:
                p.requires_grad = True
        return self

    def eval(self):
        self.training = False
        for p in self._parameters.values():
            if p is not None:
                p.requires_grad = False
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad.zero_()
