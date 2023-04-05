from __future__ import annotations

import abc
import inspect
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type

import torch._ops

import torch.onnx
import torch.onnx._internal.fx.function_dispatcher as function_dispatcher
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter
from torch.onnx._internal.fx import fx_exporter
from torch.utils import _pytree as pytree

# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276


def _replace_tuple_with_list(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    _type = list if spec.type == tuple else spec.type
    return pytree.TreeSpec(
        _type, spec.context, list(map(_replace_tuple_with_list, spec.children_specs))
    )


def _assert_identical_pytree_spec(
    spec1: pytree.TreeSpec, spec2: pytree.TreeSpec, msg: str
) -> None:
    # NOTE: The check is relaxed since dynamo returns 'list' instead of 'tuple' for outputs.
    if spec1 != spec2 and _replace_tuple_with_list(spec1) != _replace_tuple_with_list(
        spec2
    ):
        raise ValueError(f"{msg}\nExpect {spec1}.\nActual {spec2}.")


class BindInputStep(exporter.InputFormatStep):
    def __init__(self, model_signature: inspect.Signature):
        self._model_signature = model_signature

    def format(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        # We hope the input kwargs will be mapped to bound.args after binding.
        # If not, we will raise an error.
        bound = self._model_signature.bind(*model_args, **model_kwargs)
        bound.apply_defaults()

        # keyword-only arguments are not handled.
        # bound.kwargs only contains keyword-only arguments after calling
        # bind & apply_defaults, so we raise if it's not empty.
        assert not bound.kwargs
        return (), bound.arguments


class MergeKwargsIntoArgsStep(exporter.InputFormatStep):
    def format(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        return tuple(model_args) + tuple(model_kwargs.values()), {}


class RemoveNoneInputStep(exporter.InputFormatStep):
    def format(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        assert not model_kwargs
        return tuple(arg for arg in model_args if arg is not None), {}


class FlattenInputWithTreeSpecValidationStep(exporter.InputFormatStep):
    _spec: Optional[pytree.TreeSpec] = None

    def format(
        self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        flattened_args, spec = pytree.tree_flatten((model_args, model_kwargs))
        if self._spec is None:
            self._spec = spec
        else:
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                msg="Model inputs incompatible with the format that was exported. ",
            )
        return flattened_args, {}


class FlattenOutputWithTreeSpecValidationStep(exporter.OutputFormatStep):
    _spec: Optional[pytree.TreeSpec] = None

    def format(self, model_outputs: Any) -> Sequence[Any]:
        flattened_outputs, spec = pytree.tree_flatten(model_outputs)
        if self._spec is None:
            self._spec = spec
        else:
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                msg="Model outputs incompatible with the format that was exported. ",
            )
        return flattened_outputs


class FXGraphModuleExporter(exporter.Exporter, abc.ABC):
    @property
    def decomposition_table(self) -> Mapping[torch._ops.OpOverload, Callable]:
        return function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE

    def _apply_input_format_step(
        self,
        format_step_cls: Type[exporter.InputFormatStep],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        step_init_args = step_init_args or ()
        format_step = format_step_cls(*step_init_args)
        self._input_formatter.append_step(format_step)
        return format_step.format(model_args, model_kwargs)

    def _apply_output_format_step(
        self,
        format_step_cls: Type[exporter.OutputFormatStep],
        model_outputs: Any,
        step_init_args: Optional[Sequence[Any]] = None,
    ) -> Any:
        step_init_args = step_init_args or ()
        format_step = format_step_cls(*step_init_args)
        self._output_formatter.append_step(format_step)
        return format_step.format(model_outputs)

    @_beartype.beartype
    def export_fx_to_onnx(
        self,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ) -> torch.onnx.ExportOutput:
        # Apply decomposition table to the input graph.
        decomposed_module = passes.Decompose(
            fx_module,
            self.decomposition_table,
            enable_dynamic_axes=self.options.dynamic_shapes,
        ).run(*fx_module_args)

        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            onnxscript_graph = passes.export_fx_to_onnxscript(
                decomposed_module, self.options
            )

        # ONNX does not support None inputs. During graph building, all None inputs are
        # removed. Here we register this step to input formatter.
        self._apply_input_format_step(
            fx_exporter.RemoveNoneInputStep, fx_module_args, {}
        )

        # Export TorchScript graph to ONNX ModelProto.
        onnx_model = onnxscript_graph.to_model_proto(self.options.opset_version)
        return torch.onnx.ExportOutput(
            onnx_model, self._input_formatter, self._output_formatter
        )
