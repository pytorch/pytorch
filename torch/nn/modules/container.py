from collections import OrderedDict
import string
import torch
import warnings, random
from .module import Module


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    """A sequential container.
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
    """Holds submodules in a list.
    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.
    Arguments:
        modules (list, optional): a list of modules to add
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
        self._ordering = []
        if modules is not None:
            self += modules

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._modules[self._ordering[idx]]

    def __setitem__(self, idx, module):
        return setattr(self, self._ordering[idx], module)

    def __delitem__(self, idx):
        delattr(self, self._ordering[idx])
        del self._ordering[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for x in self._ordering:
            yield self._modules[x]

    def __iadd__(self, modules):
        return self.extend(modules)

    def append(self, module):
        """Appends a given module at the end of the list.
        Arguments:
            module (nn.Module): module to append
        """
        newhash = self._make_new_hash()
        self._ordering.append(newhash)
        self.add_module(newhash, module)
        return self

    def insert(self, i, module):
        """ Inserts a given module at a given position.
        Arguments:
            i (int): position where to insert
            module (nn.Module): module to insert
        """
        newhash = self._make_new_hash()
        self._ordering.insert(i, newhash)
        self.add_module(newhash, module)
        return self

    def extend(self, modules):
        """Appends modules from a Python list at the end.
        Arguments:
            modules (list): list of modules to append
        """
        if not isinstance(modules, list):
            raise TypeError("ModuleList.extend should be called with a "
                            "list, but got " + type(modules).__name__)
        for i, module in enumerate(modules):
            newhash = self._make_new_hash()
            self._ordering.append(newhash)
            self.add_module(newhash, module)
        return self

    def _make_new_hash(self):
        newhash = str(random.getrandbits(32))
        while newhash in self._modules:
            newhash = str(random.getrandbits(32))
        return newhash



class ParameterList(Module):
    """Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (list, optional): a list of :class:`nn.Parameter`` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """Appends parameters from a Python list at the end.

        Arguments:
            parameters (list): list of parameters to append
        """
        if not isinstance(parameters, list):
            raise TypeError("ParameterList.extend should be called with a "
                            "list, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self
