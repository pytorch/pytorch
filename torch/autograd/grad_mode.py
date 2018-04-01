import torch


class no_grad(object):
    r"""Context-manager that disabled gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.
    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.

    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> with torch.no_grad():
        ...   y = x * 2
        >>> y.requires_grad
        False
    """

    def __init__(self):
        self.prev = torch.is_grad_enabled()

    def __enter__(self):
        torch._C.set_grad_enabled(False)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False


class enable_grad(object):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation inside a :class:`~no_grad` context. This has
    no effect outside of :class:`~no_grad`.


    Example::

        >>> x = torch.tensor([1], requires_grad=True)
        >>> with torch.no_grad():
        ...   with torch.enable_grad():
        ...     y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad

    """

    def __init__(self):
        self.prev = torch.is_grad_enabled()

    def __enter__(self):
        torch._C.set_grad_enabled(True)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False


class set_grad_enabled(object):
    r"""Context-manager that sets gradient calculation to on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

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
        >>> set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        True

    """

    def __init__(self, mode):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(mode)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        return False
