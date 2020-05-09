import torch
import functools
import inspect

class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            while True:
                try:
                    with self:
                        x = next(gen)
                    yield x
                except StopIteration:
                    break
        return generator_context


class no_grad(_DecoratorContextManager):
    r"""Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.

    This mode has no effect when using :class:`~enable_grad` context manager .

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
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
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(False)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False


class enable_grad(_DecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator. (Make sure to instantiate with parenthesis.)


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> with torch.no_grad():
        ...   with torch.enable_grad():
        ...     y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True

    """
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(True)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False


class set_grad_enabled(object):
    r"""Context-manager that sets gradient calculation to on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    When using :class:`~enable_grad` context manager, :class:`~set_grad_enabled(False)`
    has no effect.

    This context manager is thread local; it will not affect computation
    in other threads.

    Arguments:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...   y = x * 2
        >>> y.requires_grad
        False
        >>> torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(mode)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False
