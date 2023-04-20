from __future__ import annotations

import abc
import inspect
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

import torch._ops
import torch.fx

import torch.onnx
import torch.onnx._internal.fx.function_dispatcher as function_dispatcher
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter
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


class FXGraphModuleExporter(exporter.Exporter, abc.ABC):
    _input_adapter: exporter.InputAdapter
    _output_adapter: exporter.OutputAdapter

    @property
    def decomposition_table(self) -> Mapping[torch._ops.OpOverload, Callable]:
        return function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE

    def _apply_input_adapt_step(
        self,
        adapt_step_cls: Type[exporter.InputAdaptStep],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Apply an input adapt step to the model args and kwargs.

        An input adapt step object is initialized, applied and recorded as part of
        ``self._input_adapter`.

        Args:
            adapt_step_cls: The input adapt step class.
            model_args: The model args.
            model_kwargs: The model kwargs.
            step_init_args: The input adapt step initialization arguments.

        Returns:
            The adapted model args and kwargs.
        """
        step_init_args = step_init_args or ()
        adapt_step = adapt_step_cls(*step_init_args)
        self._input_adapter.append_step(adapt_step)
        return adapt_step.apply(model_args, model_kwargs)

    def _apply_output_adapt_step(
        self,
        adapt_step_cls: Type[exporter.OutputAdaptStep],
        model_outputs: Any,
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Any:
        """Apply an output adapt step to the model outputs.

        An output adapt step object is initialized, applied and recorded as part of
        ``self._output_adapter`.

        Args:
            adapt_step_cls: The output adapt step class.
            model_outputs: The model outputs.
            step_init_args: The input adapt step initialization arguments.

        Returns:
            The adapted model outputs.
        """
        step_init_args = step_init_args or ()
        adapt_step = adapt_step_cls(*step_init_args)
        self._output_adapter.append_step(adapt_step)
        return adapt_step.apply(model_outputs)

    @_beartype.beartype
    def export_fx_to_onnx(
        self,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ) -> torch.onnx.ExportOutput:
        # Apply decomposition table to the input graph.
        module = passes.Decompose(
            fx_module,
            self.decomposition_table,
            enable_dynamic_axes=self.options.dynamic_shapes,
        ).run(*fx_module_args)

        # ONNX does not support views and mutations.
        # Functionalize to get a semantically equivalent graph without mutations.
        module = passes.Functionalize(
            module, enable_dynamic_axes=self.options.dynamic_shapes
        ).run(*fx_module_args)
        # Input mutations are detected and distilled after `Functionalize` pass.
        # Remove them since ONNX inference does not need them.
        module = passes.RemoveInputMutation(module).run(*fx_module_args)

        # Run ShapeInferenceWithFakeTensor to get static shape of nodes for op_level_debug purposes
        # The pass added nodes with static shape into original node metadata:
        # node.meta["static_shape"]: FakeTensor/int/float/SymInt/SynFloat
        if self.options.op_level_debug:
            module = passes.ShapeInferenceWithFakeTensor(module).run(*fx_module_args)

        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            onnxscript_graph = passes.export_fx_to_onnxscript(module, self.options)
            # ONNX does not support None inputs. During graph building, all None inputs
            # are removed. Here we register this step to input adapter.
            self._apply_input_adapt_step(RemoveNoneInputStep, fx_module_args, {})
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            self._output_adapter.append_step(FlattenOutputStep())

        # Export TorchScript graph to ONNX ModelProto.
        onnx_model = onnxscript_graph.to_model_proto(self.options.opset_version)
        return torch.onnx.ExportOutput(
            onnx_model,
            self._input_adapter,
            self._output_adapter,
            self.options.diagnostics_context,
        )
