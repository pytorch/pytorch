from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from torch.utils import _pytree as pytree

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
