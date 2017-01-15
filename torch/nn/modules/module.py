from itertools import chain
from collections import OrderedDict
import functools

import torch
from ..backends.thnn import backend as thnn_backend
from ..parameter import Parameter
from torch.autograd import Variable
import torch.utils.hooks as hooks


class Module(object):
    """Base class for all Modules defined in the nn package.

    Even the Container class derives from it.
    """
    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
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

    def _apply(self, fn):
        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param.grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def apply(self, fn):
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
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter) or (params and name in params):
            self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self, destination=None, prefix=''):
        """Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Example:
            >>> print(module.state_dict().keys())
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in chain(self._buffers.items(), self._parameters.items()):
            if param is not None:
                destination[prefix + name] = param
        return destination

    def load_state_dict(self, state_dict, prefix=''):
        """Replaces module parameters using values from a given state_dict.

        This will load all values from the state dict (including such that
        weren't registered before loading).

        Arguments:
            state_dict (dict): A dict containing loaded parameters and
                persistent buffers.
        """
        for name, param in self._parameters.items():
            new_param = state_dict.get(prefix + name, param)
            if not isinstance(new_param, Parameter) and new_param is not None:
                raise TypeError(
                    "expected torch.autograd.Parameter for key '{}' (got {})"
                    .format(prefix + name, torch.typename(new_param)))
            self._parameters[name] = new_param
        for name, buf in self._buffers.items():
            self._buffers[name] = state_dict.get(prefix + name, buf)

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

    def children(self):
        """Returns an iterator over children modules."""
        if False:
            yield

    def modules(self, memo=None):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self

    def train(self):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        self.training = True
        return self

    def eval(self):
        """Sets the module in evaluation mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        self.training = False
        return self

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            p.grad.data.zero_()

    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())
