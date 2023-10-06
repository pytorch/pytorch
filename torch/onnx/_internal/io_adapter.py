from __future__ import annotations

import inspect

from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)

import torch

from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree

# TODO(bowbao): Add diagnostics for IO adapters.


@runtime_checkable
class InputAdaptStep(Protocol):
    """A protocol that defines a step in the input adapting process.

    The input adapting process is a sequence of steps that are applied to the
    PyTorch model inputs to transform them into the inputs format expected by the
    exported ONNX model. Each step takes the PyTorch model inputs as arguments and
    returns the transformed inputs.

    This serves as a base formalized construct for the transformation done to model
    input signature by any individual component in the exporter.
    """

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        ...


class InputAdapter:
    """A class that adapts the PyTorch model inputs to exported ONNX model inputs format."""

    def __init__(self, steps: Optional[List[InputAdaptStep]] = None):
        self._steps = steps or []

    @_beartype.beartype
    def append_step(self, step: InputAdaptStep) -> None:
        """Appends a step to the input adapt steps.

        Args:
            step: The step to append.
        """
        self._steps.append(step)

    @_beartype.beartype
    def apply(
        self, *model_args, **model_kwargs
    ) -> Sequence[Union[int, float, bool, str, "torch.Tensor", None]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Args:
            model_args: The PyTorch model inputs.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.
        """
        args: Sequence[Any] = model_args
        kwargs: Mapping[str, Any] = model_kwargs
        for step in self._steps:
            args, kwargs = step.apply(args, kwargs)
        assert not kwargs
        return args


@runtime_checkable
class OutputAdaptStep(Protocol):
    """A protocol that defines a step in the output adapting process.

    The output adapting process is a sequence of steps that are applied to the
    PyTorch model outputs to transform them into the outputs format produced by the
    exported ONNX model. Each step takes the PyTorch model outputs as arguments and
    returns the transformed outputs.

    This serves as a base formalized construct for the transformation done to model
    output signature by any individual component in the exporter.
    """

    def apply(self, model_outputs: Any) -> Any:
        ...


class OutputAdapter:
    """A class that adapts the PyTorch model outputs to exported ONNX model outputs format."""

    def __init__(self, steps: Optional[List[OutputAdaptStep]] = None):
        self._steps = steps or []

    @_beartype.beartype
    def append_step(self, step: OutputAdaptStep) -> None:
        """Appends a step to the output format steps.

        Args:
            step: The step to append.
        """
        self._steps.append(step)

    @_beartype.beartype
    def apply(
        self, model_outputs: Any
    ) -> Sequence[Union["torch.Tensor", int, float, bool, str]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Args:
            model_outputs: The PyTorch model outputs.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
        """
        for step in self._steps:
            model_outputs = step.apply(model_outputs)
        return model_outputs


# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276


def _replace_tuple_with_list(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    _type = list if spec.type == tuple else spec.type
    return pytree.TreeSpec(
        _type, spec.context, list(map(_replace_tuple_with_list, spec.children_specs))
    )


def _open_top_level_list_if_single_element(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    if spec.type == list and len(spec.children_specs) == 1:
        return spec.children_specs[0]
    return spec


def _assert_identical_pytree_spec(
    spec1: pytree.TreeSpec, spec2: pytree.TreeSpec, error_message: str
) -> None:
    """Assert the two `TreeSpec` objects are identical.

    Args:
        spec1: The first `TreeSpec` object.
        spec2: The second `TreeSpec` object.
        error_message: The error message to raise if the two `TreeSpec` objects are not
            identical.

    Raises:
        ValueError: If the two `TreeSpec` objects are not identical.
    """
    # TODO(bowbao): Turn this check into diagnostic. Consider warning instead of error.
    pass_if_any_checks: Sequence[Callable[[], bool]] = [
        lambda: spec1 == spec2,
        # FIXME: Bug in `dynamo.export`. Sometimes outputs returned in 'list' instead of 'tuple'.
        lambda: _replace_tuple_with_list(spec1) == _replace_tuple_with_list(spec2),
        # FIXME: Bug in `dynamo.export`. Sometimes single function return is wrapped in list.
        lambda: _open_top_level_list_if_single_element(spec1) == spec2,
        lambda: spec1 == _open_top_level_list_if_single_element(spec2),
    ]

    if not any(check() for check in pass_if_any_checks):
        raise ValueError(f"{error_message}\nExpect {spec1}.\nActual {spec2}.")


class BindInputStep:
    """Bind the input arguments to the model signature."""

    def __init__(self, model_signature: inspect.Signature):
        self._model_signature = model_signature

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Bind the input arguments to the model signature.

        We hope the input kwargs will be mapped to bound.args after binding.
        If not, we will raise an error.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args and kwargs. args is always empty.

        Raises:
            ValueError: If there are keyword-only arguments left after binding args and
                kwargs to model signature.
        """
        bound = self._model_signature.bind(*model_args, **model_kwargs)
        bound.apply_defaults()

        # keyword-only arguments are not handled.
        # bound.kwargs only contains keyword-only arguments after calling
        # bind & apply_defaults, so we raise if it's not empty.
        if bound.kwargs:
            raise ValueError("Keyword-only arguments are not supported.")
        return (), bound.arguments


class MergeKwargsIntoArgsStep:
    """Merge the input kwargs into the input args."""

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Merge the input kwargs into the input args.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args and kwargs. kwargs is always empty.
        """
        return tuple(model_args) + tuple(model_kwargs.values()), {}


class LiftParametersAndBuffersIntoArgsStep:
    """Append parameters and buffers to model's positional argument list."""

    def __init__(self, inputs: Tuple["torch.Tensor", ...]) -> None:
        self.inputs = inputs

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Append model's parameters and buffers into its input.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args + appended inputs and kwargs.
        """
        return (*model_args, *self.inputs), model_kwargs


class ConvertComplexToRealRepresentationInputStep:
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Convert complex tensors to float tensors.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args and kwargs.
        """
        return (
            tuple(
                torch.view_as_real(arg)
                if isinstance(arg, torch.Tensor) and arg.is_complex()
                else arg
                for arg in model_args
            ),
            model_kwargs,
        )


class RemoveNoneInputStep:
    """Remove `None` from arguments.

    This adapt step assumes ``model_kwargs`` is empty. It also assumes ``model_args``
    is flattened, i.e. it does not check `None` inside nested collections.
    """

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Remove `None` from arguments.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args and kwargs.

        Raises:
            ValueError: If `model_kwargs` is not empty.
        """
        assert not model_kwargs
        return tuple(arg for arg in model_args if arg is not None), {}


class RemoveNonTensorInputStep:
    """Remove the non-tensor input arguments.

    Dynamo does not support non-tensor input arguments (https://github.com/pytorch/pytorch/issues/99534).

    Specifically, it does put the input into graph with an empty node, but consumed by no ones.
    The concrete value is embedded into the graph as a constant arg of a target node. Meta
    suggests in this case that one should rewrite the model code to make it tensor if the
    input value is supposed to change at runtime. We might need to further investigate
    the feasibility of that suggestion.

    For example,

        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        x = torch.randn(1, 1, 2, dtype=torch.float32)
        gm_fun, _ = dynamo.export(func, x, b=8.0, aten_graph=True, tracing_mode="real")

        # class GraphModule(torch.nn.Module):
        #     def forward(self, x, b):
        #         arg0: f32[1, 1, 2], arg1, = fx_pytree.tree_flatten_spec(([x, b], {}), self._in_spec)
        #         # File: path/to/pytorch/test_constant_input.py:5, code: y = x + b
        #         add_tensor: f32[1, 1, 2] = torch.ops.aten.add.Tensor(arg0, 8.0);  arg0 = None

        #         # File: path/to/pytorch/test_constant_input.py:6, code: z = y.relu()
        #         relu_default: f32[1, 1, 2] = torch.ops.aten.relu.default(add_tensor)
        #         return pytree.tree_unflatten([add_tensor, relu_default], self._out_spec)

    Empty torch.fx.Node input leading to a mismatched number of input with PyTorch, as
    it's ignored in ONNX graph. Thus, we delete the useless input here.

    """

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Remove Constant from arguments.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the model args and kwargs.

        Raises:
            ValueError: If `model_kwargs` is not empty.
        """
        assert not model_kwargs
        return (
            tuple(
                arg
                for arg in model_args
                if not isinstance(arg, (int, float, bool, str))
            ),
            {},
        )


class FlattenInputWithTreeSpecValidationStep:
    """Flatten nested collection types and return a flat list of elements.

    ONNX can't represent collection types (e.g., dictionary, tuple of tuple of tensor,
    etc).

    This class stores the `SpecTree` output produced when `adapt` was called the first
    time. It then validates the `SpecTree` output produced from later `adapt` calls.
    """

    _spec: Optional[pytree.TreeSpec] = None

    def apply(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Flatten the model args and kwargs and validate the `SpecTree` output.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.

        Returns:
            A tuple of the flattened model args and kwargs. The kwargs is empty, because
            they are flattened and merged into the args.

        Raises:
            ValueError: If the `SpecTree` output produced from the current `model_outputs`
                is not identical to the `SpecTree` output produced from the first
                `model_outputs` that was passed to this method.
        """
        flattened_args, spec = pytree.tree_flatten((model_args, model_kwargs))
        if self._spec is None:
            self._spec = spec
        else:
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                error_message="Model inputs incompatible with the format that was exported. ",
            )
        return flattened_args, {}


class FlattenOutputStep:
    """Flatten nested collection types and return a flat list of elements.

    ONNX can't represent collection types (e.g., dictionary, tuple of tuple of tensor,
    etc).

    NOTE: Ideally we would want to use ``FlattenOutputWithTreeSpecValidationStep``, such
    that `SpecTree` can be validate for new model outputs. However, this is not possible
    currently because we never have access to real PyTorch model outputs during export.
    Only traced outputs may be available, but they are not an accurate reflection of the
    original PyTorch model outputs format as they are typically in their own unique format,
    depending on the tracing strategy.
    """

    def apply(self, model_outputs: Any) -> Sequence[Any]:
        """Flatten the model outputs."""
        flattened_outputs, _ = pytree.tree_flatten(model_outputs)
        return flattened_outputs


class ConvertComplexToRealRepresentationOutputStep:
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(self, model_outputs: Any) -> Any:
        """Convert float tensors to complex tensors.

        Args:
            model_output: The model output.

        Returns:
            A tuple of the model output.
        """
        return [
            torch.view_as_real(output)
            if isinstance(output, torch.Tensor) and torch.is_complex(output)
            else output
            for output in model_outputs
        ]


class FlattenOutputWithTreeSpecValidationStep:
    """Same as ``FlattenOutputStep``, with additional `TreeSpec` validation.

    This class stores the `SpecTree` output produced when `adapt` was called the first
    time. It then validates the `SpecTree` output produced from later `adapt` calls.
    """

    _spec: Optional[pytree.TreeSpec] = None

    def apply(self, model_outputs: Any) -> Sequence[Any]:
        """Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.

        Returns:
            flattened_outputs: The flattened model outputs.

        Raises:
            ValueError: If the `SpecTree` output produced from the current `model_outputs`
                is not identical to the `SpecTree` output produced from the first
                `model_outputs` that was passed to this method.
        """
        flattened_outputs, spec = pytree.tree_flatten(model_outputs)
        if self._spec is None:
            self._spec = spec
        else:
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                error_message="Model outputs incompatible with the format that was exported. ",
            )
        return flattened_outputs
