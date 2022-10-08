import sys
import torch
import functools
import inspect
from typing import Any, Callable, TypeVar, cast

__all__ = ['no_grad', 'enable_grad', 'set_grad_enabled',
           'inference_mode', 'set_multithreading_enabled']


# Used for annotating the decorator usage of 'no_grad' and 'enable_grad'.
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, func: F) -> F:
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return func(*args, **kwargs)
        return cast(F, decorate_context)

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # Generators are suspended and unsuspended at `yield`, hence we
            # make sure the grad mode is properly set every time the execution
            # flow returns into the wrapped generator and restored when it
            # returns through our `yield` to our caller (see PR #49017).
            try:
                # Issuing `None` to a generator fires it up
                with self.clone():
                    response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        request = yield response

                    except GeneratorExit:
                        # Inform the still active generator about its imminent closure
                        with self.clone():
                            gen.close()
                        raise

                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        with self.clone():
                            response = gen.throw(*sys.exc_info())

                    else:
                        # Pass the last request to the generator and get its response
                        with self.clone():
                            response = gen.send(request)

            # We let the exceptions raised above by the generator's `.throw` or
            # `.send` methods bubble up to our caller, except for StopIteration
            except StopIteration as e:
                # The generator informed us that it is done: take whatever its
                # returned value (if any) was and indicate that we're done too
                # by returning it (see docs for python's return-statement).
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class no_grad(_DecoratorContextManager):
    r"""Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

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
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
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


class enable_grad(_DecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

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
        ...   with torch.enable_grad():
        ...     y = x * 2
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
        ...   y = x * 2
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
        torch._C._set_grad_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)

    def clone(self):
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context-manager that enables or disables inference mode

    InferenceMode is a new context manager analogous to :class:`~no_grad`
    to be used when you are certain your operations will have no interactions
    with autograd (e.g., model training). Code run under this mode gets better
    performance by disabling view tracking and version counter bumps. Note that
    unlike some other mechanisms that locally enable or disable grad,
    entering inference_mode also disables to :ref:`forward-mode AD <forward-mode-ad>`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    .. note::
        Inference mode is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Args:
        mode (bool): Flag whether to enable or disable inference mode

    Example::
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...   y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isnt quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...   return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False

    """
    def __init__(self, mode=True):
        if not torch._jit_internal.is_scripting():
            super().__init__()
        # Holds a python binding to a RAII guard that can enable or disable
        # inference mode
        self._inference_mode_raii_guard = None
        self.mode = mode

    def __enter__(self):
        self._inference_mode_raii_guard = torch._C._InferenceMode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del self._inference_mode_raii_guard

    def clone(self):
        return self.__class__(self.mode)


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
        self.mode = mode
        self.multithreadeding_enabled_guard = torch._C._MultithreadingEnabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        del self.multithreadeding_enabled_guard

    def clone(self):
        return self.__class__(self.mode)
