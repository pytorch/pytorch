import torch
import os
from .grad_mode import _DecoratorContextManager
from collections import namedtuple

from typing import Any

__all__ = ["UnpackedDualTensor", "enter_dual_level", "exit_dual_level", "make_dual", "unpack_dual", "dual_level"]

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
    r"""Associates a tensor value with a forward gradient, the tangent, to create a
    "dual tensor", which is used to compute forward AD gradients.
    The result is a new tensor aliased to :attr:`tensor` with :attr:`tangent` embedded
    as an attribute as-is if it has the same storage layout or copied otherwise.
    The tangent attribute can be recovered with :func:`unpack_dual`.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows one to compute the Jacobian-vector product (`jvp`)
    between `J` and a given vector `v` as follows.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.

    """
    # See NOTE: [forward-mode AD decompositions mechanism]
    #
    # Import from torch._decomp import decompositions_for_jvp to register
    # decompositions for jvp to the jit registry
    #
    # FIXME: We specify that __debug__ must be True because
    # if python is run with -OO or -O flags (i.e., __debug__ is False), we encounter the
    # following error:
    #
    # Return value was annotated as having type Tuple[NoneType, NoneType] but is actually of
    # type Tuple[Tensor, Tensor]:
    #   File ".../torch/_decomp/__init__.py", line 1585
    #     else:
    #         buffer = z
    #     return min - torch.log1p(z), buffer
    #     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    if os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__:
        from torch._decomp import decompositions_for_jvp  # noqa: F401

    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError("Trying to create a dual Tensor for forward AD but no level "
                           "exists, make sure to enter_dual_level() first.")
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(f"Expected primal to be floating point or complex, but got: {tensor.dtype}")
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(f"Expected tangent to be floating point or complex, but got: {tangent.dtype}")

    return torch._VF._make_dual(tensor, tangent, level=level)

_UnpackedDualTensor = namedtuple('_UnpackedDualTensor', ['primal', 'tangent'])


class UnpackedDualTensor(_UnpackedDualTensor):
    r"""Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.
    See :func:`unpack_dual` for more details."""
    pass


def unpack_dual(tensor, *, level=None):
    r"""Unpacks a "dual tensor" to get both its Tensor value and its forward AD gradient.
    The result is a namedtuple ``(primal, tangent)`` where ``primal`` is a view of
    :attr:`tensor`'s primal and ``tangent`` is :attr:`tensor`'s tangent as-is.
    Neither of these tensors can be dual tensor of level :attr:`level`.

    This function is backward differentiable.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)
        ...     jvp = unpack_dual(out).tangent

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    if level is None:
        level = _current_level

    if level < 0:
        return UnpackedDualTensor(tensor, None)

    primal, dual = torch._VF._unpack_dual(tensor, level=level)

    return UnpackedDualTensor(primal, dual)


class dual_level(_DecoratorContextManager):
    r"""Context-manager that enables forward AD. All forward AD computation must
    be performed in a ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()

# Private helper functions
_is_fwd_grad_enabled = torch._C._is_fwd_grad_enabled

# Private helper function to enable or disable fwd grad.
# If you're a user and want to use this, please file an issue to discuss the use case.
class _set_fwd_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = _is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_fwd_grad_enabled(self.prev)
