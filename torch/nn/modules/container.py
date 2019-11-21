import warnings
from collections import OrderedDict
from torch._six import container_abcs
from itertools import islice
import operator

import torch
from .module import Module
from torch._jit_internal import _copy_to_script_wrapper


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

    To make it easier to understand, here is a small example::

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

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

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

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ModuleDict` (the argument to :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules):
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, container_abcs.Mapping):
            if isinstance(modules, (OrderedDict, ModuleDict)):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

    def forward(self):
        raise NotImplementedError()


class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be indexed like a regular Python
    list, but parameters it contains are properly registered, and will be
    visible by all :class:`~torch.nn.Module` methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter` to add

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

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._parameters.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        idx = self._get_abs_string_index(idx)
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ParameterDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ParameterDict` (the argument to
      :meth:`~torch.nn.ParameterDict.update`).

    Note that :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter`) or an iterable of key-value pairs
            of type (string, :class:`~torch.nn.Parameter`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self.register_parameter(key, parameter)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.keys())

    def __contains__(self, key):
        return key in self._parameters

    def clear(self):
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()

    def pop(self, key):
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()

    def items(self):
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self):
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters):
        r"""Update the :class:`~torch.nn.ParameterDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, container_abcs.Mapping):
            if isinstance(parameters, (OrderedDict, ParameterDict)):
                for key, parameter in parameters.items():
                    self[key] = parameter
            else:
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
