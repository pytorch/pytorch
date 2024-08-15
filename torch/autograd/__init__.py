# mypy: allow-untyped-defs
"""
``torch.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

import warnings
from typing import cast, List, Optional, Sequence, Tuple, Union

import torch
from torch import _vmap_internals
from torch.overrides import handle_torch_function, has_torch_function, is_tensor_like
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge

from . import forward_ad, functional, graph
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .function import Function, NestedIOFunction
from .grad_mode import (
    _force_original_view_tracking,
    _unsafe_preserve_version_counter,
    enable_grad,
    inference_mode,
    no_grad,
    set_grad_enabled,
    set_multithreading_enabled,
)
from .gradcheck import gradcheck, gradgradcheck
from .graph import _engine_run_backward
from .variable import Variable


__all__ = [
    "Variable",
    "Function",
    "backward",
    "grad_mode",
    "NestedIOFunction",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
    "variable",
]

_OptionalTensor = Optional[torch.Tensor]
_ShapeorNestedShape = Union[_size, Sequence[_size], torch.Tensor]


def _calculate_shape(
    output: Union[torch.Tensor, graph.GradientEdge],
    grad: torch.Tensor,
    is_grads_batched: bool,
) -> Tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    # is_same_size ensures that both tensors are either nested or non nested
    # circular import
    from torch.nested._internal.nested_tensor import NestedTensor

    if isinstance(output, graph.GradientEdge):
        # We have already checked that we are not a C++ NestedTensor
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with GradientEdge")
        out_metadata = output.node._input_metadata[output.output_nr]
        return torch.Size(out_metadata.shape), grad.shape

    if output.is_nested and not isinstance(output, NestedTensor):
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()

        return out_shape, grad_shape

    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


def _make_grads(
    outputs: Union[Sequence[torch.Tensor], Sequence[graph.GradientEdge]],
    grads: Sequence[_OptionalTensor],
    is_grads_batched: bool,
) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        out = cast(Union[torch.Tensor, graph.GradientEdge], out)
        out_size = None
        out_device = None

        if isinstance(out, graph.GradientEdge):
            out_metadata = out.node._input_metadata[out.output_nr]
            out_size = torch.Size(out_metadata.shape)
            out_dtype = out_metadata.dtype
            out_device = out_metadata.device
            out_is_nested = out_metadata.is_nested_tensor
            if out_metadata.is_cpp_nested_tensor:
                raise RuntimeError(
                    "C++ NestedTensor are not supported with GradientEdge"
                )
            out_is_cpp_nested = False
        else:
            # circular import
            from torch.nested._internal.nested_tensor import NestedTensor

            assert isinstance(out, torch.Tensor)
            out_dtype = out.dtype
            out_is_nested = out.is_nested
            out_is_cpp_nested = out_is_nested and not isinstance(out, NestedTensor)
            if not out_is_cpp_nested:
                out_size = out.shape

        if isinstance(grad, torch.Tensor):
            from torch.fx.experimental.symbolic_shapes import expect_true, sym_eq

            first_grad = grad if not is_grads_batched else grad[0]

            # TODO: We can remove this conditional once we uniformly use
            # singleton int to represent jagged dimension, so that size() call
            # on nested tensor works.
            if out_is_cpp_nested:
                assert isinstance(out, torch.Tensor)
                shape_matches = torch.is_same_size(out, first_grad)
            else:
                # We need to do a regular size check, without going through
                # the operator, to be able to handle unbacked symints
                # (expect_true ensures we can deal with unbacked)
                assert out_size is not None
                shape_matches = expect_true(sym_eq(out_size, first_grad.size()))

            if not shape_matches:
                out = cast(Union[torch.Tensor, graph.GradientEdge], out)
                out_shape, grad_shape = _calculate_shape(
                    out, first_grad, is_grads_batched
                )
                if is_grads_batched:
                    raise RuntimeError(
                        "If `is_grads_batched=True`, we interpret the first "
                        "dimension of each grad_output as the batch dimension. "
                        "The sizes of the remaining dimensions are expected to match "
                        "the shape of corresponding output, but a mismatch "
                        "was detected: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + ". "
                        "If you only want some tensors in `grad_output` to be considered "
                        "batched, consider using vmap."
                    )
                else:
                    raise RuntimeError(
                        "Mismatch in shape: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + "."
                    )
            if out_dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError(
                    "For complex Tensors, both grad_output and output"
                    " are required to have the same dtype."
                    " Mismatch in dtype: grad_output["
                    + str(grads.index(grad))
                    + "] has a dtype of "
                    + str(grad.dtype)
                    + " and output["
                    + str(outputs.index(out))
                    + "] has a dtype of "
                    + str(out_dtype)
                    + "."
                )
            new_grads.append(grad)
        elif grad is None:
            if isinstance(out, graph.GradientEdge) or out.requires_grad:  # type: ignore[attr-defined]
                if isinstance(out, graph.GradientEdge):
                    assert out_size is not None
                    out_numel_is_1 = all(o == 1 for o in out_size)
                else:
                    assert isinstance(out, torch.Tensor)
                    out_numel_is_1 = out.numel() == 1
                if not out_numel_is_1:
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs"
                    )
                if not out_dtype.is_floating_point:
                    msg = (
                        "grad can be implicitly created only for real scalar outputs"
                        f" but got {out_dtype}"
                    )
                    raise RuntimeError(msg)
                if isinstance(out, graph.GradientEdge):
                    assert out_size is not None
                    assert out_device is not None
                    new_grads.append(
                        torch.ones(
                            out_size,
                            dtype=out_dtype,
                            device=out_device,
                        )
                    )
                else:
                    assert isinstance(out, torch.Tensor)
                    new_grads.append(
                        torch.ones_like(out, memory_format=torch.preserve_format)
                    )
            else:
                new_grads.append(None)
        else:
            raise TypeError(
                "gradients can be either Tensors or None, but got "
                + type(grad).__name__
            )
    return tuple(new_grads)


def _tensor_or_tensors_to_tuple(
    tensors: Optional[_TensorOrTensors], length: int
) -> Tuple[_OptionalTensor, ...]:
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    return tuple(tensors)


def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
) -> None:
    r"""Compute the sum of gradients of given tensors with respect to graph leaves.

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
        inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge], optional): Inputs w.r.t. which the gradient
            be will accumulated into ``.grad``. All other Tensors will be ignored. If
            not provided, the gradient is accumulated into all the leaf Tensors that
            were used to compute the :attr:`tensors`.
    """
    if torch._C._are_functorch_transforms_active():
        raise RuntimeError(
            "backward() called inside a functorch transform. This is not "
            "supported, please use functorch.grad or functorch.vjp instead "
            "or call backward() outside of functorch transforms."
        )

    if grad_variables is not None:
        warnings.warn(
            "`grad_variables` is deprecated. Use `grad_tensors` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError(
                "`grad_tensors` and `grad_variables` (deprecated) "
                "arguments both passed to `backward()`. Please only "
                "use `grad_tensors`."
            )
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("`inputs` argument to `backward()` cannot be empty.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
    inputs = (
        (inputs,)
        if isinstance(inputs, (torch.Tensor, graph.GradientEdge))
        else tuple(inputs)
        if inputs is not None
        else ()
    )

    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment below is that
    # some Python versions print out the first line of a multi-line function
    # calls in the traceback and some print out the last line
    _engine_run_backward(
        tensors,
        grad_tensors_,
        retain_graph,
        create_graph,
        inputs,
        allow_unreachable=True,
        accumulate_grad=True,
    )


def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: Optional[bool] = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> Tuple[torch.Tensor, ...]:
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
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
        outputs (sequence of Tensor or GradientEdge): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
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
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
        materialize_grads (bool, optional): If ``True``, set the gradient for unused inputs
            to zero instead of None. This is useful when computing higher-order derivatives.
            If ``materialize_grads`` is ``True`` and ``allow_unused`` is ``False``, an error
            will be raised. Defaults to ``False``.

    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    if is_tensor_like(outputs) or isinstance(outputs, graph.GradientEdge):
        outputs = cast(
            Union[Sequence[torch.Tensor], Sequence[graph.GradientEdge]], (outputs,)
        )
    else:
        outputs = tuple(outputs)
    if is_tensor_like(inputs) or isinstance(inputs, graph.GradientEdge):
        inputs = cast(_TensorOrTensorsOrGradEdge, (inputs,))
    else:
        inputs = tuple(inputs)
    t_outputs = tuple(i for i in outputs if is_tensor_like(i))
    t_inputs = tuple(i for i in inputs if is_tensor_like(i))
    overridable_args = t_outputs + t_inputs
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
            is_grads_batched=is_grads_batched,
            materialize_grads=materialize_grads,
        )

    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(outputs))
    grad_outputs_ = _make_grads(
        outputs, grad_outputs_, is_grads_batched=is_grads_batched
    )

    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat the same comment several times below is because
    # some Python versions print out the first line of multi-line function
    # calls in the traceback and some print out the last line
    if is_grads_batched:

        def vjp(gO):
            return _engine_run_backward(
                outputs,
                gO,
                retain_graph,
                create_graph,
                inputs,
                allow_unused,
                accumulate_grad=False,
            )

        result = _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(
            grad_outputs_
        )
    else:
        result = _engine_run_backward(
            outputs,
            grad_outputs_,
            retain_graph,
            create_graph,
            inputs,
            allow_unused,
            accumulate_grad=False,
        )
    if materialize_grads:
        if any(
            result[i] is None and not is_tensor_like(inputs[i])
            for i in range(len(inputs))
        ):
            raise RuntimeError(
                "materialize_grads cannot be used when the given input is a GradientEdge"
            )
        result = tuple(
            output
            if output is not None
            else torch.zeros_like(input, requires_grad=True)
            for (output, input) in zip(result, inputs)
        )
    return result


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


def variable(*args, **kwargs):  # noqa: D103
    raise RuntimeError(
        "torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead"
    )


# Monkey patching variable.Variable to fix FX codegen. FX generates a call by roughly doing
# f"{fn.__module__}.{fn.__name__}(...). This yields torch.autograd.variable.Variable(...) in the
# output of an FX graph.  Unfortunately the module name torch.autograd.variable is shadowed by the
# deprecated function - variable(...).
variable.Variable = Variable  # type: ignore[attr-defined]

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")

# Import all native method/classes
from torch._C._autograd import (
    _add_metadata_json,
    _disable_profiler,
    _disable_profiler_legacy,
    _enable_profiler,
    _enable_profiler_legacy,
    _enable_record_function,
    _get_sequence_nr,
    _kineto_step,
    _KinetoEvent,
    _pop_saved_tensors_default_hooks,
    _prepare_profiler,
    _profiler_enabled,
    _ProfilerResult,
    _push_saved_tensors_default_hooks,
    _record_function_with_args_enter,
    _record_function_with_args_exit,
    _set_empty_test_observer,
    _supported_activities,
    _toggle_collection_dynamic,
    DeviceType,
    kineto_available,
    ProfilerEvent,
    SavedTensor,
)
from torch._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState

from . import profiler


def _register_py_tensor_class_for_device(device, cls):
    if not isinstance(cls, type):
        raise RuntimeError("cls isn't a typeinfo object")
    torch._C._register_py_class_for_device(device, cls)


is_multithreading_enabled = torch._C._is_multithreading_enabled
torch._C._add_docstr(
    is_multithreading_enabled, "Returns True if multithreading is currently enabled."
)

is_view_replay_enabled = torch._C._is_view_replay_enabled
torch._C._add_docstr(
    is_view_replay_enabled, "Returns True if view-replay is currently enabled."
)
