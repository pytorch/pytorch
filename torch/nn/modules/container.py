from collections import OrderedDict, Iterable
import string
import torch
import warnings
from .module import Module


def _addPrefix(s_, prefix):
    if prefix is None:
        return s_
    if len(s_) == 0:
        return prefix
    else:
        return prefix + "." + s_


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class ModuleList(Module):
    r"""Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            items = self._modules.items()
            if not isinstance(items, list):
                items = list(items)
            return items[idx][1]
        else:
            return self._modules[idx]

    def __setitem__(self, idx, module):
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module, prefix=None):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module or tuple (str, nn.Module)): module to append
        """
        if isinstance(module, tuple):
            k, module = module
            module_key = _addPrefix(str(k), prefix)
        else:
            module_key = _addPrefix(str(len(self)), prefix)
        self.add_module(module_key, module)
        return self

    def extend(self, modules, prefix=None):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, Iterable) or isinstance(modules, str):
            raise TypeError("ModuleList.extend should be called with a "
                            "non-string iterable, but got " +
                            type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            if isinstance(module, tuple):
                k, module = module
                module_key = _addPrefix(str(k), prefix)
            else:
                module_key = _addPrefix(str(len(self)), prefix)
            self.add_module(module_key, module)
        return self


class ParameterList(Module):
    r"""Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter`` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            items = self._parameters.items()
            if not isinstance(items, list):
                items = list(items)
            return items[idx][1]
        else:
            return self._parameters[idx]

    def __setitem__(self, idx, param):
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def append(self, parameter, prefix=None):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        if isinstance(parameter, tuple):
            k, parameter = parameter
            param_key = _addPrefix(str(k), prefix)
        else:
            param_key = _addPrefix(str(len(self)), prefix)
        self.register_parameter(param_key, parameter)
        return self

    def extend(self, parameters, prefix=None):
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, Iterable) or isinstance(parameters, str):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            if isinstance(param, tuple):
                k, param = param
                param_key = _addPrefix(str(k), prefix)
            else:
                param_key = _addPrefix(str(offset + i), prefix)
            self.register_parameter(param_key, param)
        return self

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(\n'
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else \
                ' (GPU {})'.format(p.get_device())
            parastr = '[Parameter ({}) of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            tmpstr = tmpstr + '  (' + k + '): ' + parastr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
