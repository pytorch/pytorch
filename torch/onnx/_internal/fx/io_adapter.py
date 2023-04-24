from __future__ import annotations

import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Type,
    Union,
)

# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276

import torch

import torch.fx

import torch.onnx
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree


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

    _input_adapt_steps: List[InputAdaptStep]

    def __init__(self, input_adapt_steps: Optional[List[InputAdaptStep]] = None):
        self._input_adapt_steps = input_adapt_steps or []

    @_beartype.beartype
    def append_step(self, step: InputAdaptStep) -> None:
        """Appends a step to the input adapt steps.

        Args:
            step: The step to append.
        """
        self._input_adapt_steps.append(step)

    @_beartype.beartype
    def apply(self, *model_args, **model_kwargs) -> Sequence[torch.Tensor]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Args:
            model_args: The PyTorch model inputs.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.
        """
        args: Sequence[Any] = model_args
        kwargs: Mapping[str, Any] = model_kwargs
        for step in self._input_adapt_steps:
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

    _output_adapt_steps: List[OutputAdaptStep]

    def __init__(self, output_adapt_steps: Optional[List[OutputAdaptStep]] = None):
        self._output_adapt_steps = output_adapt_steps or []

    @_beartype.beartype
    def append_step(self, step: OutputAdaptStep) -> None:
        """Appends a step to the output format steps.

        Args:
            step: The step to append.
        """
        self._output_adapt_steps.append(step)

    @_beartype.beartype
    def apply(self, model_outputs: Any) -> Sequence[torch.Tensor]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Args:
            model_outputs: The PyTorch model outputs.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
        """
        for step in self._output_adapt_steps:
            model_outputs = step.apply(model_outputs)
        return model_outputs


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


class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""

    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self):
        self._extensions = {}
        # Register PyTree extension for HuggingFace model output.
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            pytree._register_pytree_node(class_type, flatten_func, unflatten_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)

    @_beartype.beartype
    def register_pytree_node(
        self,
        class_type: Type,
        flatten_func: pytree.FlattenFunc,
        unflatten_func: pytree.UnflattenFunc,
    ):
        """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
        assert (
            class_type not in pytree.SUPPORTED_NODES
            and class_type not in self._extensions
        ), "PyTree node already registered"
        self._extensions[class_type] = (flatten_func, unflatten_func)

    def _register_huggingface_model_output_extension(self):
        try:
            from transformers import modeling_outputs  # type: ignore[import]
        except ImportError as e:
            return

        @_beartype.beartype
        def model_output_flatten(
            output: modeling_outputs.ModelOutput,
        ) -> Tuple[List[Any], pytree.Context]:
            return list(output.values()), (type(output), list(output.keys()))

        @_beartype.beartype
        def model_output_unflatten(
            values: List[Any], context: pytree.Context
        ) -> modeling_outputs.ModelOutput:
            output_type, keys = context
            return output_type(**dict(zip(keys, values)))

        # All 'ModelOutput' subclasses are defined under module 'modeling_outputs'.
        named_model_output_classes = inspect.getmembers(
            modeling_outputs,
            lambda x: inspect.isclass(x)
            and issubclass(x, modeling_outputs.ModelOutput),
        )

        for _, class_type in named_model_output_classes:
            self.register_pytree_node(
                class_type, model_output_flatten, model_output_unflatten
            )


class DynamoFlattenOutputStep(FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`FlattenOutputStep` to support flattening arbitrary
    types via pytree extension. By default this supports many common user defined python
    types such as :class:`ModelOutput` from HuggingFace transformers.

    The pytree extension can be customized by passing in a ``_PyTreeExtensionContext``
    object. See :meth:`_PyTreeExtensionContext.register_pytree_node`.
    """

    def __init__(
        self, pytree_extension_context: Optional[_PyTreeExtensionContext] = None
    ):
        super().__init__()
        self._pytree_extension_context = (
            pytree_extension_context or _PyTreeExtensionContext()
        )

    def apply(self, model_outputs: Any) -> Sequence[Any]:
        """Flatten the model outputs, under the context of pytree extension."""
        with self._pytree_extension_context:
            return super().apply(model_outputs)


def _wrap_model_with_output_adapter(
    model: Union[torch.nn.Module, Callable],
    output_adapter: DynamoFlattenOutputStep,
) -> Callable:
    """Wrap model with output adapter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output adapter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The adapting logic is controlled by ``output_adapter``.

    Args:
        model: PyTorch model or function.
        output_adapter: Output adapter to apply to model output.
    Returns:
        Wrapped model.
    """
    model_func = model.forward if isinstance(model, torch.nn.Module) else model

    # Preserve original function signature.
    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        return output_adapter.apply(model_func(*args, **kwargs))

    return wrapped
