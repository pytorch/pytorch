"""
``torch.autograd`` provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
"""
import torch
import warnings

from .variable import Variable
from .function import Function, NestedIOFunction
from .gradcheck import gradcheck, gradgradcheck
from .grad_mode import no_grad, enable_grad, set_grad_enabled
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from . import profiler

__all__ = ['Variable', 'Function', 'backward', 'grad_mode']


def _make_grads(outputs, grads):
    new_grads = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                new_grads.append(torch.ones_like(out))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads)


def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None):
    r"""Computes the sum of gradients of given tensors w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains gradient of
    the differentiated function w.r.t. corresponding tensors (``None`` is an
    acceptable value for all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        tensors (sequence of Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (sequence of (Tensor or None)): Gradients w.r.t.
            each element of corresponding tensors. None values can be specified for
            scalar Tensors or ones that don't require grad. If a None value would
            be acceptable for all grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)

    if grad_tensors is None:
        grad_tensors = [None] * len(tensors)
    elif isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    else:
        grad_tensors = list(grad_tensors)

    grad_tensors = _make_grads(tensors, grad_tensors)
    if retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True)  # allow_unreachable flag


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
         only_inputs=True, allow_unused=False):
    r"""Computes and returns the sum of gradients of outputs w.r.t. the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the pre-computed gradients w.r.t. each of the outputs. If an
    output doesn't require_grad, then the gradient can be ``None``).

    If ``only_inputs`` is ``True``, the function will only return a list of gradients
    w.r.t the specified inputs. If it's ``False``, then gradient w.r.t. all remaining
    leaves will still be computed, and will be accumulated into their ``.grad``
    attribute.

    Arguments:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): Gradients w.r.t. each output.
            None values can be specified for scalar Tensors or ones that don't require
            grad. If a None value would be acceptable for all grad_tensors, then this
            argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (bool, optional): If ``False``, specifying inputs that were not
            used when computing outputs (and therefore their grad is always zero)
            is an error. Defaults to ``False``.
    """
    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now "
                      "(defaults to True). To accumulate gradient for other "
                      "parts of the graph, please use torch.autograd.backward.")

    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)
    elif isinstance(grad_outputs, torch.Tensor):
        grad_outputs = [grad_outputs]
    else:
        grad_outputs = list(grad_outputs)

    grad_outputs = _make_grads(outputs, grad_outputs)
    if retain_graph is None:
        retain_graph = create_graph

    return Variable._execution_engine.run_backward(
        outputs, grad_outputs, retain_graph, create_graph,
        inputs, allow_unused)


# This function applies in case of gradient checkpointing for memory
# optimization. Currently, for gradient checkpointing, we only support imperative
# backwards call i.e. torch.autograd.backward() and the torch.autograd.grad() won't
# work. The reason being that: torch.autograd.grad() only calculates the grads
# for the inputs that are passed by user but it doesn't calculate grad for
# anything else e.g. model parameters like weights, bias etc. However, for
# torch.autograd.backward(), we would actually compute the grad for the weights as well.
#
# This function returns whether the checkpointing is valid i.e. torch.autograd.backward
# or not i.e. torch.autograd.grad. The implementation works by maintaining a thread
# local variable in torch/csrc/autograd/engine.cpp which looks at the FunctionTask
# in the stack and before a FunctionTask is executed in evaluate_function, it
# checks for whether reentrant backwards is imperative or not.
# See https://github.com/pytorch/pytorch/pull/4594 for more discussion/context
def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()


def variable(*args, **kwargs):
    warnings.warn("torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead")
    return torch.tensor(*args, **kwargs)


if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")
