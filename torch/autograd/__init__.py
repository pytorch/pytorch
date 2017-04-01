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
    grad_variables = tuple(var if isinstance(var, Variable) or var is None
                           else Variable(var, volatile=True)
                           for var in grad_variables)
    Variable._execution_engine.run_backward(
        tuple(variables), grad_variables, retain_variables)


def differentiate(outputs, grad_outputs, inputs, only_inputs=True, retain_variables=True):
    """Computes and returns the sum of gradients of outputs w.r.t. the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the pre-computed gradients w.r.t. each of the outputs. If an
    output doesn't require_grad, then the gradient can be ``None``).
    Gradients can be given as Tensors when one doesn't need the graph of the
    derivative, or as Variables, in which case the graph will be created.

    If ``only_inputs`` is True, the function will only return a list of gradients
    w.r.t the specified inputs. If it's False, then gradient w.r.t. all remaining
    leaves will still be computed, and will be accumulated into their ``.grad``
    attribute.

    Arguments:
        outputs (sequence of Variable): outputs of the differentiated function.
        grad_outputs (sequence of Tensor or Variable): Gradients w.r.t each output.
            The jacobian will be multiplied by these vectors from the left.
        inputs (sequence of Variable): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        only_inputs (bool, optional): If True, gradient w.r.t. leaves that are
            part of the graph, but are not in ``inputs`` won't be computed and
            accumulated.
        retain_variables (bool, optional): If True, buffers necessary for
            computing the gradients won't be freed after use. It is only
            necessary to specify True if you want to differentiate any subgraph
            again.
    """
    grad_outputs = tuple(var if isinstance(var, Variable) or var is None
                         else Variable(var, volatile=True)
                         for var in grad_outputs)
    return Variable._execution_engine.run_backward(
        tuple(outputs), grad_outputs, retain_variables,
        tuple(inputs), only_inputs)

assert torch._C._autograd_init()
