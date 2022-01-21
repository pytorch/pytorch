import torch
from .grad_mode import _DecoratorContextManager
from collections import namedtuple

from typing import Any

# Global variable used to make the python API simpler to use
_current_level = -1

def enter_dual_level():
    r"""Function that can be used to enter a new forward grad level.
    This level can be used to make and unpack dual Tensors to compute
    forward gradients.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    new_level = torch._C._enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError("Entering a new forward AD level but the current level "
                           "is not valid. Make sure you did not modified it directly.")
    _current_level = new_level
    return new_level

def exit_dual_level(*, level=None):
    r"""Function that can be used to exit a forward grad level.
    This function deletes all the gradients associated with this
    level. Only deleting the latest entered level is allowed.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError("Trying to exit a forward AD level that was not the last one "
                           "that was created. This is not supported.")
    torch._C._exit_dual_level(level=level)
    _current_level = level - 1

def make_dual(tensor, tangent, *, level=None):
    r"""Function that creates a "dual object" that can be used to compute forward AD gradients
    based on the given Tensor and its tangent. It returns a new Tensor that shares memory with
    :attr:`tensor` and the :attr:`tangent` is used as-is.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows to compute the jacobian vector product,
    named `jvp`, between `J` and a given vector `v` as follows.

    Example::

        >>> with dual_level():
        ...   inp = make_dual(x, v)
        ...   out = f(inp)
        ...   y, jvp = unpack_dual(out)

    Please see our `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.

    """
    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError("Trying to create a dual Tensor for forward AD but no level "
                           "exists, make sure to enter_dual_level() first.")

    return torch._VF._make_dual(tensor, tangent, level=level)

UnpackedDualTensor = namedtuple('UnpackedDualTensor', ['primal', 'tangent'])

def unpack_dual(tensor, *, level=None):
    r"""Unpacks a "dual object" to return a namedtuple ``(primal, tangent)`` where
    ``primal`` is a view of :attr:`tensor`'s primal and ``tangent`` is
    :attr:`tensor`'s tangent. Neither of these tensors can be dual tensor of level
    :attr:`level`.

    This function is backward differentiable.

    Example::

        >>> with dual_level():
        ...   inp = make_dual(x, x_t)
        ...   out = f(inp)
        ...   y, jvp = unpack_dual(out)
        ...   jvp = unpack_dual(out).tangent

    Please see our `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    if level is None:
        level = _current_level

    if level < 0:
        return UnpackedDualTensor(tensor, None)

    primal, dual = torch._VF._unpack_dual(tensor, level=level)

    return UnpackedDualTensor(primal, dual)

class dual_level(_DecoratorContextManager):
    r"""Context-manager that enables forward ad. All forward AD computation must
    be performed in a ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward ad level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward ad level is supported. To compute higher-order
        forward grads, one can use `functorch <https://github.com/pytorch/functorch>`__.

    Example::

        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...   inp = make_dual(x, x_t)
        ...   # Do computations with inp
        ...   out = your_fn(inp)
        ...   _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see our `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    def __init__(self):
        super().__init__()

    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()
