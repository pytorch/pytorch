"""
``torch.autograd`` provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""
import torch
import warnings

from torch.types import _TensorOrTensors
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .variable import Variable
from .function import Function, NestedIOFunction
from .gradcheck import gradcheck, gradgradcheck
from .grad_mode import no_grad, enable_grad, set_grad_enabled, inference_mode
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from ..overrides import has_torch_function, handle_torch_function
from . import functional
from . import forward_ad
from . import graph

__all__ = ['Variable', 'Function', 'backward', 'grad_mode']

_OptionalTensor = Optional[torch.Tensor]

def _make_grads(outputs: Sequence[torch.Tensor], grads: Sequence[_OptionalTensor]) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            if not out.shape == grad.shape:
                raise RuntimeError("Mismatch in shape: grad_output["
                                   + str(grads.index(grad)) + "] has a shape of "
                                   + str(grad.shape) + " and output["
                                   + str(outputs.index(out)) + "] has a shape of "
                                   + str(out.shape) + ".")
            if out.dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError("For complex Tensors, both grad_output and output"
                                   " are required to have the same dtype."
                                   " Mismatch in dtype: grad_output["
                                   + str(grads.index(grad)) + "] has a dtype of "
                                   + str(grad.dtype) + " and output["
                                   + str(outputs.index(out)) + "] has a dtype of "
                                   + str(out.dtype) + ".")
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads)


def _tensor_or_tensors_to_tuple(tensors: Optional[_TensorOrTensors], length: int) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None, ) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors, )
    return tuple(tensors)


def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensors] = None,
) -> None:
    r"""Computes the sum of gradients of given tensors with respect to graph
    leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    ``.grad`` attributes or set them to ``None`` before calling it.
    See :ref:`Default gradient layouts<default-grad-layouts>`
    for details on the memory layout of accumulated gradients.

    .. note::
        Using this method with ``create_graph=True`` will create a reference cycle
        between the parameter and its gradient which can cause a memory leak.
        We recommend using ``autograd.grad`` when creating the graph to avoid this.
        If you have to use this function, make sure to reset the ``.grad`` fields of your
        parameters to ``None`` after use to break the cycle and avoid the leak.

    .. note::

        If you run any forward ops, create ``grad_tensors``, and/or call ``backward``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        When ``inputs`` are provided and a given input is not a leaf,
        the current implementation will call its grad_fn (even though it is not strictly needed to get this gradients).
        It is an implementation detail on which the user should not rely.
        See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    Args:
        tensors (Sequence[Tensor] or Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): The "vector" in
            the Jacobian-vector product, usually gradients w.r.t. each element of
            corresponding tensors. None values can be specified for scalar Tensors or
            ones that don't require grad. If a None value would be acceptable for all
            grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Sequence[Tensor] or Tensor, optional): Inputs w.r.t. which the gradient
            be will accumulated into ``.grad``. All other Tensors will be ignored. If
            not provided, the gradient is accumulated into all the leaf Tensors that
            were used to compute the attr::tensors.
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("'inputs' argument to backward() cannot be empty.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else \
        tuple(inputs) if inputs is not None else tuple()

    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_)
    if retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        tensors, grad_tensors_, retain_graph, create_graph, inputs,
        allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag


def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False
) -> Tuple[torch.Tensor, ...]:
    r"""Computes and returns the sum of gradients of outputs with respect to
    the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in Jacobian-vector product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    Args:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the Jacobian-vector product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
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
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    overridable_args = outputs + inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
        )

    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now "
                      "(defaults to True). To accumulate gradient for other "
                      "parts of the graph, please use torch.autograd.backward.")

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(outputs))
    grad_outputs_ = _make_grads(outputs, grad_outputs_)

    if retain_graph is None:
        retain_graph = create_graph

    return Variable._execution_engine.run_backward(
        outputs, grad_outputs_, retain_graph, create_graph,
        inputs, allow_unused, accumulate_grad=False)


# This function applies in case of gradient checkpointing for memory
# optimization. Currently, gradient checkpointing is supported only if the
# execution engine is invoked through torch.autograd.backward() and its
# inputs argument is not passed. It is not supported for torch.autograd.grad().
# This is because if inputs are specified, the gradient won't be calculated for
# anything else e.g. model parameters like weights, bias etc.
#
# This function returns whether the checkpointing is valid i.e. torch.autograd.backward
# or not i.e. torch.autograd.grad. The implementation works by maintaining a thread
# local variable in torch/csrc/autograd/engine.cpp which looks at the NodeTask
# in the stack and before a NodeTask is executed in evaluate_function, it
# checks for whether reentrant backwards is imperative or not.
# See https://github.com/pytorch/pytorch/pull/4594 for more discussion/context
def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()


def variable(*args, **kwargs):
    warnings.warn("torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead")
    return torch.tensor(*args, **kwargs)

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")

# Import all native method/classes
from torch._C._autograd import (DeviceType, ProfilerActivity, ProfilerState, ProfilerConfig, ProfilerEvent,
                                _enable_profiler_legacy, _disable_profiler_legacy, _profiler_enabled,
                                _enable_record_function, _set_empty_test_observer, kineto_available,
                                _supported_activities, _add_metadata_json, SavedTensor,
                                _register_saved_tensors_default_hooks, _reset_saved_tensors_default_hooks)

from torch._C._autograd import (_ProfilerResult, _KinetoEvent,
                                _prepare_profiler, _enable_profiler, _disable_profiler)

from . import profiler
