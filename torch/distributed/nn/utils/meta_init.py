# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional

import torch
from torch import Tensor
from torch._C import _DisableTorchDispatch  # type: ignore[attr-defined]
from torch.nn.modules import Module
from torch.utils._python_dispatch import enable_python_mode

_tls = threading.local()

# Used to check nested `init_meta()` calls.
_tls.in_call = False


@contextmanager
def _no_dispatch() -> Iterator[None]:
    """Temporarily disables the Python dispatch mode."""
    guard = _DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def _handle_arange(func, args, kwargs):
    kwargs["device"] = torch.device("cpu")

    # Although not ideal we first instantiate a CPU tensor and then immediately
    # convert it to a meta tensor. Since `arange()` is typically used for small
    # auxiliary vectors, in most cases this isn't a problem.
    return torch.empty_like(func(*args, **kwargs), device="meta")


def _handle_tril(func, args, kwargs):
    if args and isinstance(args[0], Tensor):
        return torch.empty_like(args[0], device="meta")

    return NotImplemented


class _MetaContext(Tensor):
    """Represents the Python dispatcher to be used with `enable_python_mode()`.

    Note:
        A known limitation of the Python dispatch API is that it requires the
        dispatcher to derive from `Tensor` or from one of its subclassses. In
        our implementation `_MetaContext` is a subclass of `Tensor` solely to
        fulfill this requirement.
    """

    _op_handlers: Dict[Callable, Callable] = {}

    @classmethod
    def _ensure_handlers_initialized(cls) -> None:
        # The `torch.ops` module is only available once the `torch` module is
        # fully imported; therefore, we lazy initialize our handlers.
        if cls._op_handlers:
            return

        cls._op_handlers.update(
            {
                torch.ops.aten.arange: _handle_arange,
                torch.ops.aten.tril: _handle_tril,
            }
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        cls._ensure_handlers_initialized()

        op_handler: Optional[Callable]

        # Some operators such as `tril()` do not support the meta backend. For
        # such operators we use special handlers.
        try:
            op_handler = cls._op_handlers[func]
        except KeyError:
            op_handler = None

        with _no_dispatch():
            if op_handler:
                result = op_handler(func, args, kwargs)
                # If for some reason the operator handler could not handle the
                # dispatch (e.g. due to unexpected arguments), we fall back to
                # our common handler below; otherwise, we return the result of
                # the handler.
                if result is not NotImplemented:
                    return result

            # We use a simple heuristic. If the function call has a device
            # argument, we simply override it with the meta device.
            if "device" in kwargs:
                kwargs["device"] = torch.device("meta")

            return func(*args, **kwargs)


def init_meta(module_fn: Callable[..., Module], *args, **kwargs) -> Module:
    """Constructs a module on the meta device for inspection purposes.

    This function is meant to be used if the size of a module is too big to fit
    on a single machine and you want to inspect it without instantiating.

    Internally ``init_meta()`` forces all parameters, buffers, and other tensors
    within the scope of the module and its sub-modules to use the meta device
    regardless of the actual device of the tensor, even if that device was
    explicitly passed to the constructor of the tensor. The returned module can
    be used for inspection purposes and afterwards its ``materialize()`` method
    can be called to construct a fully-instantiated module (see the example
    below).

    However note that ``init_meta()`` uses a "best effort" algorithm and is not
    guaranteed to succeed if the underlying module's implementation cannot be
    mapped to the meta device. If you are the module author, you can use the
    ``is_meta_init()`` function described below to find out whether your module
    is being instantiated in the scope of an ``init_meta()`` call and use an
    alternate logic if required:

    ::

        class MyModule(Module):
            def __init__(self):
                if torch.distributed.nn.utils.is_meta_init():
                    self.myparam = torch.empty([10,10], device="meta")
                else:
                    self.myparam = load_myparam()

    Args:
        module_fn (Callable[..., Module]):
            The constructor or the factory function of the module.
        args:
            The positional arguments to pass to ``module_fn()``.
        kwargs:
            The keyword arguments to pass to ``module_fn()``.

    Returns:
        A ``torch.nn.Module`` instance on the meta device.

    :Example:
        >>> import torch
        >>> import torch.distributed.nn
        >>> m = torch.distributed.nn.utils.init_meta(torch.nn.Linear, 5, 1)
        >>> m.weight
        Parameter containing:
        tensor(..., device='meta', requires_grad=True)
        >>> m = m.materialize()
        >>> m.weight
        Parameter containing:
        tensor([[-1.4677e+24,  4.5915e-41,  1.4013e-45,  0.0000e+00,
                 -1.4677e+24, 4.5915e-41]], requires_grad=True)
        >>>
        >>> # `init_meta()` overrides even explicitly passed device arguments.
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.param = torch.ones([10, 10], device="cpu")
        ...
        >>> m = torch.distributed.nn.utils.init_meta(MyModule)
        >>> m.param
        tensors(..., device='meta', size=(10, 10))


    Note:
        The ``args`` and ``kwargs`` arguments must be treated as immutable by
        the module constructor since they might be used a second time if
        ``materialize()`` is called.
    """

    def create_instance() -> Module:
        return module_fn(*args, **kwargs)

    if _tls.in_call:
        module = create_instance()
    else:
        _tls.in_call = True
        try:
            with enable_python_mode(_MetaContext):
                module = create_instance()
        finally:
            _tls.in_call = False

    module.materialize = create_instance  # type: ignore[assignment]

    return module


def is_meta_init() -> bool:
    """Indicates whether the module is being instantiated by ``init_meta()``."""
    return _tls.in_call
