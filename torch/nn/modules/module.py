from itertools import chain
from collections import OrderedDict
import functools

import torch
from ..backends.thnn import backend as thnn_backend
from ..parameter import Parameter
from torch.autograd import Variable
import torch.utils.hooks as hooks


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(object):
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    dump_patches = False

    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        """Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Example:
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        self._buffers[name] = tensor

    def register_parameter(self, name, param):
        """Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.creator:
            raise ValueError(
                "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def add_module(self, name, module):
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        return self._apply(lambda t: t.cuda(device_id))

    def cpu(self, device_id=None):
        """Moves all model parameters and buffers to the CPU."""
        return self._apply(lambda t: t.cpu())

    def type(self, dst_type):
        return self._apply(lambda t: t.type(dst_type))

    def float(self):
        """Casts all parameters and buffers to float datatype."""
        return self._apply(lambda t: t.float())

    def double(self):
        """Casts all parameters and buffers to double datatype."""
        return self._apply(lambda t: t.double())

    def half(self):
        """Casts all parameters and buffers to half datatype."""
        return self._apply(lambda t: t.half())

    def register_backward_hook(self, hook):
        """Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> Tensor or None

        The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
        module has multiple inputs or outputs. The hook should not modify its
        arguments, but it can optionally return a new gradient with respect to
        input that will be used in place of :attr:`grad_input` in subsequent
        computations.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[id(handle)] = hook
        return handle

    def register_forward_hook(self, hook):
        """Registers a forward hook on the module.

        The hook will be called every time :func:`forward` computes an output.
        It should have the following signature::

            hook(module, input, output) -> None

        The hook should not modify the input or output.
        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[id(handle)] = hook
        return handle

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        var = result
        while not isinstance(var, Variable):
            var = var[0]
        creator = var.creator
        if creator is not None and len(self._backward_hooks) > 0:
            if creator._backward_hooks is None:
                creator._backward_hooks = OrderedDict()
            for hook in self._backward_hooks.values():
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                creator._backward_hooks[id(wrapper)] = wrapper
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
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self, destination=None, prefix=''):
        """Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Example:
            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def parameters(self, memo=None):
        """Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Example:
            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
        """
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                yield p
        for module in self.children():
            for p in module.parameters(memo):
                yield p

    def children(self):
        """Returns an iterator over children modules."""
        memo = set()
        for module in self._modules.values():
            if module is not None and module not in memo:
                memo.add(module)
                yield module

    def modules(self, memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self
            for module in self.children():
                for m in module.modules(memo):
                    yield m

    def train(self):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        self.training = True
        for module in self.children():
            module.train()
        return self

    def eval(self):
        """Sets the module in evaluation mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        self.training = False
        for module in self.children():
            module.eval()
        return self

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.requires_grad:
                p.grad.data.zero_()

    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
