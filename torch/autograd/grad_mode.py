from typing import Any

import torch

from torch.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
    F,
)

__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "set_multithreading_enabled",
]


class no_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @torch.no_grad
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with torch.no_grad():
        ...     a = torch.nn.Parameter(torch.rand(10))
        >>> a.requires_grad
        True
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        tensor([2.])
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True
        >>> @torch.enable_grad
        ... def tripler(x):
        ...     return x * 3
        >>> with torch.no_grad():
        ...     z = tripler(x)
        >>> z.requires_grad
        True

    """

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)


class set_grad_enabled(_DecoratorContextManager):
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch.is_grad_enabled()
        self.mode = mode
        torch._C._set_grad_enabled(mode)

    def __call__(self, orig_func: F) -> F:
        torch._C._set_grad_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)

    def clone(self) -> "set_grad_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context-manager that enables or disables inference mode.

    InferenceMode is a new context manager analogous to :class:`~no_grad`
    to be used when you are certain your operations will have no interactions
    with autograd (e.g., model training). Code run under this mode gets better
    performance by disabling view tracking and version counter bumps. Note that
    unlike some other mechanisms that locally enable or disable grad,
    entering inference_mode also disables to :ref:`forward-mode AD <forward-mode-ad>`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Args:
        mode (bool or function): Either a boolean flag whether to enable or
            disable inference mode or a Python function to decorate with
            inference mode enabled

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isnt quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @torch.inference_mode
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """

    def __init__(self, mode: bool = True) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.mode = mode

    def __new__(cls, mode=True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self._inference_mode_context = torch._C._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    def clone(self) -> "inference_mode":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


def _enter_inference_mode(mode):
    mode_context = torch._C._InferenceMode(mode)
    mode_context.__enter__()
    return mode_context


def _exit_inference_mode(mode):
    mode.__exit__(None, None, None)


class set_multithreading_enabled(_DecoratorContextManager):
    r"""Context-manager that sets multithreaded backwards on or off.

    ``set_multithreading_enabled`` will enable or disable multithreaded backwards based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable multithreaded backwards (``True``), or disable
                     (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_multithreading_enabled()
        torch._C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class _force_original_view_tracking(_DecoratorContextManager):
    r"""Context-manager that sets whether or not to always enable view-replay in autograd.

    ``set_view_replay_enabled`` will enable or disable view-replay based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    When a tensor view is mutated, the autograd engine needs to decide whether or not
    to regenerate the "updated view" by either replaying the chain of views from the updated base,
    or with a single call to as_strided.

    If set_view_replay_enabled is set to True, then autograd will always use view replay.
    Otherwise, it will fall back to its existing logic.

    Args:
        mode (bool): Flag whether to enable view-replay (``True``), or disable
                     (``False``).

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_view_replay_enabled()
        torch._C._set_view_replay_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_view_replay_enabled(self.prev)

    def clone(self):
        return self.__class__(self.mode)


class _unsafe_preserve_version_counter(_DecoratorContextManager):
    r"""DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code
    (even the ones not touched directly by the context manager)!

    Ordinarily, autograd will track mutations to tensors by incrementing it's `._version` attribute.
    This is generally important for correctness, as for example, mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients, and autograd uses the version counter to detect
    and error out in this situation.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.prev_version = tensor._version

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        torch._C._autograd._unsafe_set_version_counter(self.tensor, self.prev_version)
