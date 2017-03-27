"""
torch.autograd provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to wrap all tensors in
:class:`.Variable` objects.
"""
import torch

from .variable import Variable
from .function import Function, NestedIOFunction
from .stochastic_function import StochasticFunction
from .gradcheck import gradcheck

__all__ = ['Variable', 'Function', 'StochasticFunction', 'backward']


def backward(variables, grad_variables, retain_variables=False):
    """Computes the sum of gradients of given variables w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``variables``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, the function additionaly requires specifying ``grad_variables``.
    It should be a sequence of matching length, that containins gradient of
    the differentiated function w.r.t. corresponding variables (``None`` is an
    acceptable value for all variables that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        variables (sequence of Variable): Variables of which the derivative will be
            computed.
        grad_variables (sequence of Tensor): Gradients w.r.t. each element of
            corresponding variables. Required only for non-scalar variables that
            require gradient.
        retain_variables (bool): If ``True``, buffers necessary for computing
            gradients won't be freed after use. It is only necessary to
            specify ``True`` if you want to differentiate some subgraph multiple
            times.
    """
    Variable._execution_engine.run_backward(
        tuple(variables), tuple(grad_variables), retain_variables)

assert torch._C._autograd_init()
