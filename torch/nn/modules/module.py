# mypy: allow-untyped-defs

import functools
import inspect
import itertools
import warnings
import weakref
from collections import namedtuple, OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Optional, overload, TypeVar, Union
from typing_extensions import Self

import torch
from torch import device, dtype, Tensor
from torch._prims_common import DeviceLikeType
from torch.nn.parameter import Buffer, Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import BackwardHook, RemovableHandle


__all__ = [
    "register_module_forward_pre_hook",
    "register_module_forward_hook",
    "register_module_full_backward_pre_hook",
    "register_module_backward_hook",
    "register_module_full_backward_hook",
    "register_module_buffer_registration_hook",
    "register_module_module_registration_hook",
    "register_module_parameter_registration_hook",
    "Module",
]

_grad_t = Union[tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    __slots__ = ()

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: dict[int, Callable] = OrderedDict()


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            self.module = weakref.ref(state["module"])


r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_backward_hooks: dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: dict[int, bool] = OrderedDict()
_global_forward_hooks_with_kwargs: dict[int, bool] = OrderedDict()

_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    r"""Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    You can optionally modify the output of the module by returning a new value
    that will replace the output from the :func:`forward` function.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if with_kwargs:
        _global_forward_hooks_with_kwargs[handle.id] = True
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def register_module_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = False

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
    hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""Register a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_pre_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_pre_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    handle = RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = True

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

    dump_patches: bool = False

    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    training: bool
    _parameters: dict[str, Optional[Parameter]]
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]
    _backward_pre_hooks: dict[int, Callable]
    _backward_hooks: dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: dict[int, bool]
    # forward hooks that should always be called even if an exception is raised
    _forward_hooks_always_called: dict[int, bool]
    _forward_pre_hooks: dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: dict[int, bool]
    _state_dict_hooks: dict[int, Callable]
    _load_state_dict_pre_hooks: dict[int, Callable]
    _state_dict_pre_hooks: dict[int, Callable]
    _load_state_dict_post_hooks: dict[int, Callable]
    _modules: dict[str, Optional["Module"]]
    call_super_init: bool = False
    _compiled_call_impl: Optional[Callable] = None

    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        torch._C._log_api_usage_once("python.nn_module")

        # Backward compatibility: no args used to be allowed when call_super_init=False
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                ""
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                " given"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_backward_pre_hooks", OrderedDict())
        super().__setattr__("_backward_hooks", OrderedDict())
        super().__setattr__("_is_full_backward_hook", None)
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", {})

        if self.call_super_init:
            super().__init__(*args, **kwargs)

    forward: Callable[..., Any] = _forward_unimplemented

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(
                f"buffer name should be a string. Got {torch.typename(name)}"
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                "(torch Tensor or None required)"
            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(
                f"parameter name should be a string. Got {torch.typename(name)}"
            )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                "(torch.nn.Parameter or None required)"
            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method."
            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(
                f"module name should be a string. Got {torch.typename(name)}"
            )
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If at any point along the path resulting from
                the target string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            return self

        atoms: list[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def set_submodule(
        self, target: str, module: "Module", strict: bool = False
    ) -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        .. note::
            If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
            or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
            the method will only attempt to replace an existing submodule and throw an error if
            the submodule does not exist.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(3, 3, 3)
                    )
                    (linear): Linear(3, 3)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To override the ``Conv2d`` with a new submodule ``Linear``, you
        could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
        where ``strict`` could be ``True`` or ``False``

        To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
        you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

        In the above if you set ``strict=True`` and call
        ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
        will be raised because ``net_b`` does not have a submodule named ``conv``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.
            strict: If ``False``, the method will replace an existing submodule
                or create a new submodule if the parent module exists. If ``True``,
                the method will only attempt to replace an existing submodule and throw an error
                if the submodule doesn't already exist.

        Raises:
            ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
            AttributeError: If at any point along the path resulting from
                the ``target`` string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: list[str] = target.split(".")
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "`" + "module" + f"` is not an nn.Module, found {type(module)}"
            )
        if len(atoms) == 1:
            parent: torch.nn.Module = self
        else:
            parent_key = ".".join(atoms[:-1])
            parent = self.get_submodule(parent_key)

        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(
                parent._get_name() + " has no attribute `" + atoms[-1] + "`"
            )
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + atoms[-1] + "` is not an nn.Module")
        setattr(parent, atoms[-1], module)

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: torch.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer: torch.Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        should_use_swap_tensors = (
            torch.__future__.get_swap_module_params_on_conversion()
        )

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            p_should_use_set_data = compute_should_use_set_data(param, param_applied)

            # subclasses may have multiple child tensors so we need to use swap_tensors
            p_should_use_swap_tensors = (
                should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)
            )

            param_grad = param.grad
            if p_should_use_swap_tensors:
                try:
                    if param_grad is not None:
                        # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                        # Decrement use count of the gradient by setting to None
                        param.grad = None
                    param_applied = torch.nn.Parameter(
                        param_applied, requires_grad=param.requires_grad
                    )
                    torch.utils.swap_tensors(param, param_applied)
                except Exception as e:
                    if param_grad is not None:
                        param.grad = param_grad
                    raise RuntimeError(
                        f"_apply(): Couldn't swap {self._get_name()}.{key}"
                    ) from e
                out_param = param
            elif p_should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                g_should_use_set_data = compute_should_use_set_data(
                    param_grad, grad_applied
                )
                if p_should_use_swap_tensors:
                    grad_applied.requires_grad_(param_grad.requires_grad)
                    try:
                        torch.utils.swap_tensors(param_grad, grad_applied)
                    except Exception as e:
                        raise RuntimeError(
                            f"_apply(): Couldn't swap {self._get_name()}.{key}.grad"
                        ) from e
                    out_param.grad = param_grad
                elif g_should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param_grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(
                        param_grad.requires_grad
                    )

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))

    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.ipu(device))

    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.xpu(device))

    def mtia(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the MTIA.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing the optimizer if the module will
        live on MTIA while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.mtia(device))

    def cpu(self: T) -> T:
        r"""Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    def to_empty(
        self: T, *, device: Optional[DeviceLikeType], recurse: bool = True
    ) -> T:
        r"""Move the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.
            recurse (bool): Whether parameters and buffers of submodules should
                be recursively moved to the specified device.

        Returns:
            Module: self
        """
        return self._apply(
            lambda t: torch.empty_like(t, device=device), recurse=recurse
        )

    @overload
    def to(
        self,
        device: Optional[DeviceLikeType] = ...,
        dtype: Optional[dtype] = ...,
        non_blocking: bool = ...,
    ) -> Self:
        ...

    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self:
        ...

    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self:
        ...

    def to(self, *args, **kwargs):
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError(
                    "nn.Module.to only accepts floating point or complex "
                    f"dtypes, but got desired dtype={dtype}"
                )
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected."
                )

        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise

        return self._apply(convert)

    def register_full_backward_pre_hook(
        self,
        hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a backward pre-hook on the module.

        The hook will be called every time the gradients for the module are computed.
        The hook should have the following signature::

            hook(module, grad_output) -> tuple[Tensor] or None

        The :attr:`grad_output` is a tuple. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the output that will be used in place of :attr:`grad_output` in
        subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
        all non-Tensor arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward_pre`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward_pre`` hooks
                on this :class:`torch.nn.Module`. Note that global
                ``backward_pre`` hooks registered with
                :func:`register_module_full_backward_pre_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        handle = RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_backward_hook(
        self, hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.

        This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
        the behavior of this function will change in future versions.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is True:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        self._is_full_backward_hook = False

        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_full_backward_hook(
        self,
        hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.

        The hook will be called every time the gradients with respect to a module
        are computed, i.e. the hook will execute if and only if the gradients with
        respect to module outputs are computed. The hook should have the following
        signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
        with respect to the inputs and outputs respectively. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the input that will be used in place of :attr:`grad_input` in
        subsequent computations. :attr:`grad_input` will only correspond to the inputs given
        as positional arguments and all kwarg arguments are ignored. Entries
        in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
        arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs or outputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward`` hooks on
                this :class:`torch.nn.Module`. Note that global
                ``backward`` hooks registered with
                :func:`register_module_full_backward_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is False:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        self._is_full_backward_hook = True

        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        if prepend:
            self._backward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def _get_backward_hooks(self):
        r"""Return the backward hooks for use in the call function.

        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
        full_backward_hooks: list[Callable] = []
        if _global_is_full_backward_hook is True:
            full_backward_hooks += _global_backward_hooks.values()
        if self._is_full_backward_hook is True:
            full_backward_hooks += self._backward_hooks.values()

        non_full_backward_hooks: list[Callable] = []
        if _global_is_full_backward_hook is False:
            non_full_backward_hooks += _global_backward_hooks.values()
        if self._is_full_backward_hook is False:
            non_full_backward_hooks += self._backward_hooks.values()

        return full_backward_hooks, non_full_backward_hooks

    def _get_backward_pre_hooks(self):
        backward_pre_hooks: list[Callable] = []
        backward_pre_hooks += _global_backward_pre_hooks.values()
        backward_pre_hooks += self._backward_pre_hooks.values()

        return backward_pre_hooks

    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        if not isinstance(result, torch.Tensor):
            if not (
                isinstance(result, tuple)
                and all(isinstance(r, torch.Tensor) for r in result)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not return a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_output. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            result = (result,)

        if not isinstance(inputs, torch.Tensor):
            if not (
                isinstance(inputs, tuple)
                and all(isinstance(i, torch.Tensor) for i in inputs)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not take as input a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_input. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            inputs = (inputs,)

        # At this point we are sure that inputs and result are tuple of Tensors
        out_grad_fn = {r.grad_fn for r in result if r.grad_fn is not None}
        if len(out_grad_fn) == 0 or (
            len(out_grad_fn) == 1 and grad_fn not in out_grad_fn
        ):
            warnings.warn(
                "Using a non-full backward hook when outputs are nested in python data structure "
                "is deprecated and will be removed in future versions. This hook will be missing "
                "some grad_output.",
                FutureWarning,
                stacklevel=2,
            )
        elif len(out_grad_fn) > 1:
            warnings.warn(
                "Using a non-full backward hook when outputs are generated by different autograd Nodes "
                "is deprecated and will be removed in future versions. This hook will be missing "
                "some grad_output. Please use register_full_backward_hook to get the documented behavior.",
                FutureWarning,
                stacklevel=2,
            )
        else:
            # At this point the grad_output part of the hook will most likely be correct
            inputs_grad_fn = {i.grad_fn for i in inputs if i.grad_fn is not None}

            next_functions = {n[0] for n in grad_fn.next_functions}

            if inputs_grad_fn != next_functions:
                warnings.warn(
                    "Using a non-full backward hook when the forward contains multiple autograd Nodes "
                    "is deprecated and will be removed in future versions. This hook will be missing "
                    "some grad_input. Please use register_full_backward_hook to get the documented "
                    "behavior.",
                    FutureWarning,
                    stacklevel=2,
                )

    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, tuple[Any, ...]], Optional[Any]],
            Callable[
                [T, tuple[Any, ...], dict[str, Any]],
                Optional[tuple[Any, dict[str, Any]]],
            ],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(
            self._forward_hooks,
            extra_dict=[
                self._forward_hooks_with_kwargs,
                self._forward_hooks_always_called,
            ],
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def _slow_forward(self, *input, **kwargs):
        tracing_state = torch._C._get_tracing_state()
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # type ignore was added because at this point one knows that
            # torch.jit._trace._trace_module_map is not Optional and has type Dict[Any, Any]
            name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None  # type: ignore[index, operator] # noqa: B950
            if name:
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = self.forward(*input, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result

    def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._call_impl(*args, **kwargs)

    # torchrec tests the code consistency with the following code
    # fmt: off
    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            return forward_call(*args, **kwargs)

        result = None
        called_always_called_hooks = set()

        def inner():
            nonlocal result, args, kwargs

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                        if args_kwargs_result is not None:
                            if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple "
                                    f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = BackwardHook(self, full_backward_hooks, backward_pre_hooks)
                args = bw_hook.setup_input_hook(args)

            result = forward_call(*args, **kwargs)
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs or hook_id in _global_forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                if not isinstance(result, (torch.Tensor, tuple)):
                    warnings.warn("For backward hooks to be called,"
                                  " module output should be a Tensor or a tuple of Tensors"
                                  f" but received {type(result)}")
                result = bw_hook.setup_output_hook(result)

            # Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, torch.Tensor):
                    if isinstance(var, dict):
                        var = next(v for v in var.values() if isinstance(v, torch.Tensor))
                    else:
                        var = var[0]
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        # This is technically not behavior equivalent when compiling, but it's
        # incredibly unlikely we will ever support throwing an exception in NN
        # module, and then catching it here, and then reraising it, and then
        # catching it again, and expecting the resulting frame to be compiled.
        # The reraise here just gunks up our exception handling for no good
        # reason.  Don't try to run the always called hooks in event of
        # exception.
        if torch.compiler.is_compiling():
            return inner()

        try:
            return inner()
        except Exception:
            # run always called hooks if they have not already been run
            # For now only forward hooks have the always_call option but perhaps
            # this functionality should be added to full backward hooks as well.
            for hook_id, hook in _global_forward_hooks.items():
                if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("global module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                        else:
                            hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue
            # raise exception raised in try block
            raise
    # fmt: on

    __call__: Callable[..., Any] = _wrapped_call_impl

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Support loading old checkpoints that don't have the following attrs:
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if "_forward_pre_hooks_with_kwargs" not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_with_kwargs" not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()
        if "_is_full_backward_hook" not in self.__dict__:
            self._is_full_backward_hook = None
        if "_backward_pre_hooks" not in self.__dict__:
            self._backward_pre_hooks = OrderedDict()

    # It is crucial that the return type is not annotated as `Any`, otherwise type checking
    # on `torch.nn.Module` and all its subclasses is largely disabled as a result. See:
    # https://github.com/pytorch/pytorch/pull/115074
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                    "(torch.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                        "(torch.nn.Module or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if isinstance(value, Buffer) or buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                            "(torch.nn.Buffer, torch.Tensor or None expected)"
                        )
                    if isinstance(value, Buffer):
                        persistent = value.persistent
                    else:
                        persistent = name not in self._non_persistent_buffers_set
                    # === HACK ===
                    # This whole block below should just be:
                    # self.register_buffer(name, value, persistent)

                    # But to support subclasses of nn.Module that (wrongfully) implement a
                    # register_buffer() method that doesn't have the "persistent"
                    # argument. Only pass it in if it is accepted otherwise assume
                    # it is always true
                    if self.register_buffer is torch.nn.Module.register_buffer:
                        self.register_buffer(name, value, persistent)
                    else:
                        sign = inspect.signature(self.register_buffer)
                        if "persistent" in sign.parameters:
                            self.register_buffer(name, value, persistent)
                        else:
                            if not persistent:
                                raise RuntimeError(
                                    "Registering a non-persistent buffer "
                                    "on a Module subclass that implements "
                                    "register_buffer() without the persistent "
                                    "argument is not allowed."
                                )
                            # Assume that the implementation without the argument has the
                            # behavior from before the argument was added: persistent=True
                            self.register_buffer(name, value)
                    # === HACK END ===
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _register_state_dict_hook(self, hook):
        r"""Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None or state_dict

        The registered hooks can modify the ``state_dict`` inplace or return a new one.
        If a new ``state_dict`` is returned, it will only be respected if it is the root
        module that :meth:`~nn.Module.state_dict` is called from.
        """
        if getattr(hook, "_from_public_api", False):
            raise RuntimeError(
                "Cannot register the same function as the state dict post hook that was "
                "previously registered via register_state_dict_post_hook"
            )
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_post_hook(self, hook):
        r"""Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.
        """
        # In _register_state_dict_hook there was a bug described in
        # https://github.com/pytorch/pytorch/issues/117437 where the return value
        # was only respected for the root module but not child submodules.
        # We fix this in this public version by only allowing inplace modifications on
        # the state_dict by the hook. However, since hooks registered via both these
        # APIs will be added to `_state_dict_hooks` and the type of `_state_dict_hooks`
        # cannot be changed due to many dependencies on it, we mark a hook
        # as being registered via the public API by setting `_from_public_api` on it.
        # In the implementation of `state_dict`, if the callable does not have this
        # flag, the old behavior of respecting the return value will be preserved
        # for the root module, otherwise, we ensure that the hook returns None.
        hook._from_public_api = True
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
        handle = RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", Module.get_extra_state)
            is not Module.get_extra_state
        ):
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar("T_destination", bound=dict[str, Any])

    @overload
    def state_dict(
        self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...
    ) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]:
        ...

    # TODO: Change `*args` to `*` and remove the corresponding warning in docs when BC allows.
    # Also remove the logic for arg parsing together.
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if not getattr(hook, "_from_public_api", False):
                if hook_result is not None:
                    destination = hook_result
            else:
                if hook_result is not None:
                    raise RuntimeError("state_dict post-hook must return None")
        return destination

    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""See :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` for details.

        A subtle difference is that if ``with_module`` is set to ``False``, then the
        hook will not take the ``module`` as the first argument whereas
        :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` always takes the
        ``module`` as the first argument.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
            hook, self if with_module else None
        )
        return handle

    def register_load_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
        """
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected torch.Tensor or Tensor-like object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if (
                    not is_param_lazy
                    and len(param.shape) == 0
                    and len(input_param.shape) == 1
                ):
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        f"size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, "
                        f"the shape in current model is {param.shape}."
                    )
                    continue

                if (
                    param.is_meta
                    and not input_param.is_meta
                    and not assign_to_params_buffers
                ):
                    warnings.warn(
                        f"for {key}: copying from a non-meta parameter in the checkpoint to a meta "
                        "parameter in the current model, which is a no-op. (Did you mean to "
                        "pass `assign=True` to assign items in the state dictionary to their "
                        "corresponding key in the module instead of copying them in place?)"
                    )

                try:
                    with torch.no_grad():
                        if use_swap_tensors:
                            new_input_param = param.module_load(
                                input_param, assign=assign_to_params_buffers
                            )
                            if id(new_input_param) == id(input_param) or id(
                                new_input_param
                            ) == id(param):
                                raise RuntimeError(
                                    "module_load returned one of self or other, please .detach() "
                                    "the result if returning one of the inputs in module_load"
                                )
                            if isinstance(param, torch.nn.Parameter):
                                if not isinstance(new_input_param, torch.nn.Parameter):
                                    new_input_param = torch.nn.Parameter(
                                        new_input_param,
                                        requires_grad=param.requires_grad,
                                    )
                                else:
                                    new_input_param.requires_grad_(param.requires_grad)
                            torch.utils.swap_tensors(param, new_input_param)
                            del new_input_param
                        elif assign_to_params_buffers:
                            # Shape checks are already done above
                            if isinstance(param, torch.nn.Parameter):
                                if not isinstance(input_param, torch.nn.Parameter):
                                    input_param = torch.nn.Parameter(
                                        input_param, requires_grad=param.requires_grad
                                    )
                                else:
                                    input_param.requires_grad_(param.requires_grad)
                            setattr(self, name, input_param)
                        else:
                            param.copy_(input_param)
                except Exception as ex:
                    action = "swapping" if use_swap_tensors else "copying"
                    error_msgs.append(
                        f'While {action} the parameter named "{key}", '
                        f"whose dimensions in the model are {param.size()} and "
                        f"whose dimensions in the checkpoint are {input_param.size()}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "set_extra_state", Module.set_extra_state)
            is not Module.set_extra_state
        ):
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix) :].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When set to ``False``, the properties of the tensors
                in the current module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata["assign_to_params_buffers"] = assign
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)  # noqa: F821

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for _name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]:
        r"""Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def share_memory(self: T) -> T:
        r"""See :meth:`torch.Tensor.share_memory_`."""
        return self._apply(lambda t: t.share_memory_())

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = {}
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # type: ignore[assignment]

        return replica

    def compile(self, *args, **kwargs):
        """
        Compile this Module's forward using :func:`torch.compile`.

        This Module's `__call__` method is compiled and all arguments are passed as-is
        to :func:`torch.compile`.

        See :func:`torch.compile` for details on the arguments for this function.
        """
        self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)
