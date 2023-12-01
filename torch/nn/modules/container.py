import warnings
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator

import torch
from .module import Module
from ..parameter import Parameter
from torch._jit_internal import _copy_to_script_wrapper

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self

__all__ = ['Container', 'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict']

T = TypeVar('T', bound=Module)


# Copied from torch.nn.modules.module, required for a custom __repr__ for ModuleList
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Container(Module):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             f'of Sequential class, but {str(type(other))} is given.')

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError('add operator supports only objects '
                             f'of Sequential class, but {str(type(other))} is given.')

    def __mul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> 'Sequential':
        return self.__mul__(other)

    def __imul__(self, other: int) -> Self:
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise AssertionError(
                f'module should be of type: {Module}')
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(
                f'Index out of range: {index}')
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> 'Sequential':
        for layer in sequential:
            self.append(layer)
        return self


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        return str(idx)

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'ModuleList']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> Self:
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> 'ModuleList':
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def __repr__(self):
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + '()'

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + '('
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: Module) -> 'ModuleList':
        r"""Append a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def extend(self, modules: Iterable[Module]) -> Self:
        r"""Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    # remove forward alltogether to fallback on Module's _forward_unimplemented


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
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

    _modules: Dict[str, Module]  # type: ignore[assignment]

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict."""
        self._modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (str): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys."""
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs."""
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values."""
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with key-value pairs from a mapping, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented


class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be used like a regular Python
    list, but Tensors that are :class:`~torch.nn.Parameter` are properly registered,
    and will be visible by all :class:`~torch.nn.Module` methods.

    Note that the constructor, assigning an element of the list, the
    :meth:`~torch.nn.ParameterDict.append` method and the :meth:`~torch.nn.ParameterDict.extend`
    method will convert any :class:`~torch.Tensor` into :class:`~torch.nn.Parameter`.

    Args:
        parameters (iterable, optional): an iterable of elements to add to the list.

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, values: Optional[Iterable[Any]] = None) -> None:
        super().__init__()
        self._size = 0
        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> Any:
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        else:
            idx = self._get_abs_string_index(idx)
            return getattr(self, str(idx))

    def __setitem__(self, idx: int, param: Any) -> None:
        # Note that all other function that add an entry to the list part of
        # the ParameterList end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the list part and thus won't
        # call into this function.
        idx = self._get_abs_string_index(idx)
        if isinstance(param, torch.Tensor) and not isinstance(param, Parameter):
            param = Parameter(param)
        return setattr(self, str(idx), param)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, parameters: Iterable[Any]) -> Self:
        return self.extend(parameters)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> 'ParameterList':
        """Append a given value at the end of the list.

        Args:
            value (Any): value to append
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[Any]) -> Self:
        """Append values from a Python iterable to the end of the list.

        Args:
            values (iterable): iterable of values to append
        """
        # Tensor is an iterable but we never want to unpack it here
        if not isinstance(values, container_abcs.Iterable) or isinstance(values, torch.Tensor):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(values).__name__)
        for value in values:
            self.append(value)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f' ({p.device})'
                else:
                    device_str = ''
                parastr = '{} containing: [{} of size {}{}]'.format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    p.dtype, size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)

        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError('ParameterList should not be called.')


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but Parameters it
    contains are properly registered, and will be visible by all Module methods.
    Other objects are treated as would be done by a regular Python dictionary

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
    :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
    will preserve their ordering.

    Note that the constructor, assigning an element of the dictionary and the
    :meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
    :class:`~torch.nn.Parameter`.

    Args:
        values (iterable, optional): a mapping (dictionary) of
            (string : Any) or an iterable of key-value pairs
            of type (string, Any)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Any = None) -> None:
        super().__init__()
        self._keys: Dict[str, None] = {}
        if parameters is not None:
            self.update(parameters)

    def _key_to_attr(self, key: str) -> str:
        if not isinstance(key, str):
            raise TypeError("Index given to ParameterDict cannot be used as a key as it is "
                            f"not a string (type is '{type(key).__name__}'). Open an issue on "
                            "github if you need non-string keys.")
        else:
            # Use the key as-is so that `.named_parameters()` returns the right thing
            return key

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __setitem__(self, key: str, value: Any) -> None:
        # Note that all other function that add an entry to the dictionary part of
        # the ParameterDict end up here. So this is the only place where we need
        # to wrap things into Parameter if needed.
        # Objects added via setattr() are not in the dictionary part and thus won't
        # call into this function.
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, torch.Tensor) and not isinstance(value, Parameter):
            value = Parameter(value)
        setattr(self, attr, value)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(list(self._keys))

    def copy(self) -> 'ParameterDict':
        """Return a copy of this :class:`~torch.nn.ParameterDict` instance."""
        # We have to use an OrderedDict because the ParameterDict constructor
        # behaves differently on plain dict vs OrderedDict
        return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
        """Set the default for a key in the Parameterdict.

        If key is in the ParameterDict, return its value.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.

        Args:
            key (str): key to set default for
            default (Any): the parameter set to the key
        """
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        """Remove all items from the ParameterDict."""
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        r"""Remove key from the ParameterDict and return its parameter.

        Args:
            key (str): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> Tuple[str, Any]:
        """Remove and return the last inserted `(key, parameter)` pair from the ParameterDict."""
        k, _ = self._keys.popitem()
        # We need the key in the _keys to be able to access/del
        self._keys[k] = None
        val = self[k]
        del self[k]
        return k, val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        r"""Return the parameter associated with key if present. Otherwise return default if provided, None if not.

        Args:
            key (str): key to get from the ParameterDict
            default (Parameter, optional): value to return if key not present
        """
        return self[key] if key in self else default

    def fromkeys(self, keys: Iterable[str], default: Optional[Any] = None) -> 'ParameterDict':
        r"""Return a new ParameterDict with the keys provided.

        Args:
            keys (iterable, string): keys to make the new ParameterDict from
            default (Parameter, optional): value to set for all keys
        """
        return ParameterDict((k, default) for k in keys)

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys."""
        return self._keys.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        r"""Return an iterable of the ParameterDict key/value pairs."""
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        r"""Return an iterable of the ParameterDict values."""
        return (self[k] for k in self._keys)

    def update(self, parameters: Union[Mapping[str, Any], 'ParameterDict']) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
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
                # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                self[p[0]] = p[1]  # type: ignore[assignment]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self.items():
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f' ({p.device})'
                else:
                    device_str = ''
                parastr = '{} containing: [{} of size {}{}]'.format(
                    "Parameter" if isinstance(p, Parameter) else "Tensor",
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')

    def __or__(self, other: 'ParameterDict') -> 'ParameterDict':
        copy = self.copy()
        copy.update(other)
        return copy

    def __ror__(self, other: 'ParameterDict') -> 'ParameterDict':
        copy = other.copy()
        copy.update(self)
        return copy

    def __ior__(self, other : 'ParameterDict') -> Self:
        self.update(other)
        return self
