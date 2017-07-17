"""
torch.autograd provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to wrap all tensors in
:class:`.Variable` objects.
"""
import torch
import warnings

from .variable import Variable
from .function import Function, NestedIOFunction
from .stochastic_function import StochasticFunction
from .gradcheck import gradcheck

__all__ = ['Variable', 'Function', 'StochasticFunction', 'backward']


def _make_grads(outputs, grads, user_create_graph):
    if user_create_graph is not None:
        create_graph = user_create_graph
    else:
        create_graph = any(isinstance(grad, Variable) and not grad.volatile
                           for grad in grads)

    new_grads = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, Variable):
            new_grads.append(grad)
        elif torch.is_tensor(grad):
            new_grads.append(Variable(grad, volatile=not create_graph))
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                data = out.data
                new_grads.append(
                    Variable(data.new().resize_as_(data).fill_(1), volatile=not create_graph))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors, Variables or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads), create_graph


def backward(variables, grad_variables=None, retain_graph=None, create_graph=None, retain_variables=None):
    """Computes the sum of gradients of given variables w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``variables``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, the function additionally requires specifying ``grad_variables``.
    It should be a sequence of matching length, that contains gradient of
    the differentiated function w.r.t. corresponding variables (``None`` is an
    acceptable value for all variables that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        variables (sequence of Variable): Variables of which the derivative will be
            computed.
        grad_variables (sequence of (Tensor, Variable or None)): Gradients w.r.t.
            each element of corresponding variables.  Any tensors will be
            automatically converted to Variables that are volatile unless
            ``create_graph`` is True.  None values can be specified for scalar
            Variables or ones that don't require grad. If a None value would
            be acceptable for all grad_variables, then this argument is optional.
        retain_graph (bool, optional): If False, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to True
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If true, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to False, unless ``grad_variables`` contains at least one
            non-volatile Variable.
    """
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    if grad_variables is None:
        grad_variables = [None] * len(variables)
    elif isinstance(grad_variables, Variable) or torch.is_tensor(grad_variables):
        grad_variables = [grad_variables]
    else:
        grad_variables = list(grad_variables)

    grad_variables, create_graph = _make_grads(variables, grad_variables, create_graph)

    if retain_variables is not None:
        if retain_graph is not None:
            raise ValueError("only one of retain_graph and retain_variables can be specified")
        retain_graph = retain_variables
        warnings.warn("retain_variables option is deprecated and will be removed in 0.3. "
                      "Use retain_graph instead.")
    elif retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        variables, grad_variables, retain_graph)


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=None, only_inputs=True):
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
        inputs (sequence of Variable): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor or Variable): Gradients w.r.t. each output.
            Any tensors will be automatically converted to Variables that are
            volatile unless ``create_graph`` is True.  None values can be
            specified for scalar Variables or ones that don't require grad.
            If a None value would be acceptable for all grad_variables, then
            this argument is optional.
        retain_graph (bool, optional): If False, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to True
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If True, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to False, unless ``grad_variables`` contains at least one
            non-volatile Variable.
        only_inputs (bool, optional): If True, gradient w.r.t. leaves that are
            part of the graph, but don't appear in ``inputs`` won't be computed
            and accumulated. Defaults to True.
    """

    outputs = (outputs,) if isinstance(outputs, Variable) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, Variable) else tuple(inputs)
    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)
    elif isinstance(grad_outputs, Variable) or torch.is_tensor(grad_outputs):
        grad_outputs = [grad_outputs]
    else:
        grad_outputs = list(grad_outputs)

    grad_outputs, create_graph = _make_grads(outputs, grad_outputs, create_graph)
    if retain_graph is None:
        retain_graph = create_graph

    return Variable._execution_engine.run_backward(
        outputs, grad_outputs, retain_graph,
        inputs, only_inputs)

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")
