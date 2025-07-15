# mypy: allow-untyped-defs
# NOTE: This file is referenced by name at
#       /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES.
#       introduced by https://github.com/pytorch/pytorch/pull/98894.
#       If this file is renamed, moved, etc please update the reference there!

from __future__ import annotations

import contextlib
import inspect
from typing import Any, Callable, TYPE_CHECKING

import torch._dynamo
import torch.fx
from torch.onnx._internal import _exporter_legacy
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""

    _extensions: dict[type, tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]

    def __init__(self) -> None:
        self._extensions = {}
        # Register PyTree extension for HuggingFace model output.
        self._register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            pytree._private_register_pytree_node(
                class_type,
                flatten_func,
                unflatten_func,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)

    def register_pytree_node(
        self,
        class_type: type,
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
        if class_type in pytree.SUPPORTED_NODES or class_type in self._extensions:
            # PyTree node already registered.
            # E.g., `huggingface/transformer` registers `ModelOutput` as PyTree node after
            # https://github.com/huggingface/transformers/pull/25358.
            return
        self._extensions[class_type] = (flatten_func, unflatten_func)

    def _register_huggingface_model_output_extension(self):
        try:
            from transformers import modeling_outputs  # type: ignore[import]
        except ImportError:
            return

        def model_output_flatten(
            output: modeling_outputs.ModelOutput,
        ) -> tuple[list[Any], pytree.Context]:
            return list(output.values()), (type(output), list(output.keys()))

        def model_output_unflatten(
            values: list[Any], context: pytree.Context
        ) -> modeling_outputs.ModelOutput:
            output_type, keys = context
            return output_type(**dict(zip(keys, values)))

        # All 'ModelOutput' subclasses are defined under module 'modeling_outputs'.
        named_model_output_classes = inspect.getmembers(
            modeling_outputs,
            lambda x: (
                inspect.isclass(x)
                and issubclass(x, modeling_outputs.ModelOutput)
                and x is not modeling_outputs.ModelOutput
            ),
        )

        for _, class_type in named_model_output_classes:
            self.register_pytree_node(
                class_type,
                model_output_flatten,
                model_output_unflatten,  # type: ignore[arg-type ]
            )


class DynamoExport(_exporter_legacy.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(
        self,
        aten_graph: bool | None = None,
    ):
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(
        self,
        options: _exporter_legacy.ResolvedExportOptions,
        model: torch.nn.Module | Callable,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        # `dynamo.export` does not recognize custom user defined classes as output type.
        # Apply wrapper to adapt the outputs back to `dynamo.export` compatible types,
        # i.e. :class:`torch.Tensor`.
        wrapped_model = model

        # Translate callable to FX graph.
        #
        fake_mode = (
            options.fake_context.fake_mode
            if options.fake_context
            else contextlib.nullcontext()
        )
        fx_mode = "symbolic" if options.dynamic_shapes else "fake"
        with fake_mode:  # type: ignore[attr-defined]
            graph_module, graph_guard = torch._dynamo.export(
                wrapped_model,
                tracing_mode=fx_mode,
            )(
                *model_args,
                **model_kwargs,
            )
        del graph_guard  # Unused
        torch._dynamo.reset()

        return self.pre_export_passes(options, model, graph_module, model_args)  # type: ignore[return-value]

    def pre_export_passes(
        self,
        options: _exporter_legacy.ResolvedExportOptions,
        original_model: torch.nn.Module | Callable,
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        return _exporter_legacy.common_pre_export_passes(
            options, original_model, fx_module, fx_module_args
        )
