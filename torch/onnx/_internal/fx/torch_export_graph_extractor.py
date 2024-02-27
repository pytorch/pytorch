# NOTE: This file is referenced by name at
#       /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES.
#       introduced by https://github.com/pytorch/pytorch/pull/98894.
#       If this file is renamed, moved, etc please update the reference there!

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING, Union

import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra

if TYPE_CHECKING:
    from torch.export.exported_program import ExportedProgram


class TorchExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(
        self,
        aten_graph: Optional[bool] = None,
    ):
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(
        self,
        options: exporter.ResolvedExportOptions,
        model: "ExportedProgram",  # type: ignore[override]
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        # No need to translate callable to FX graph.
        # This FX Graph extractor assumes `model` was obtained through
        #     exported_program = torch.export.export(
        #         model,
        #         args=model_args,  # type: ignore[arg-type]
        #         kwargs=model_kwargs,  # type: ignore[arg-type]
        #     )

        # Export FX graph to ONNX ModelProto.
        self.input_adapter.append_step(
            io_adapter.FlattenInputWithTreeSpecValidationInputStep()
        )
        self.input_adapter.append_step(
            io_adapter.PrependParamsBuffersConstantAotAutogradInputStep()
        )

        # ONNX does not support None inputs. During graph building, all None inputs
        # are removed. Here we register this step to input adapter.
        options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())

        # NOTE: temp workaround for https://github.com/pytorch/pytorch/issues/99534
        # Dynamo doesn't support non-tensor inputs.
        options.fx_tracer.input_adapter.append_step(
            io_adapter.RemoveNonTensorInputStep()
        )

        # ONNX does not support complex inputs. During graph building, all complex inputs
        # are converted to real representation inputs. Here we register this step to
        # input/output adapter.
        options.fx_tracer.input_adapter.append_step(
            io_adapter.ConvertComplexToRealRepresentationInputStep()
        )

        updated_model_args = self.input_adapter.apply(
            *model_args, model=model, **model_kwargs
        )

        # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
        # tensor, etc), we flatten the collection and register each element as output.
        options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())

        # Output post-processing steps should happen after `FlattenOutputStep`.
        options.fx_tracer.output_adapter.append_step(
            io_adapter.ConvertComplexToRealRepresentationOutputStep()
        )

        options.fx_tracer.output_adapter.append_step(
            io_adapter.PrependParamsAndBuffersAotAutogradOutputStep()
        )

        # run_decomposition generates a new graph module with decomposed ops.
        # Thus, we need to run this step after io_adapters.
        model = model.run_decompositions(options.decomposition_table)

        # Export FX graph to ONNX ModelProto.
        return self.pre_export_passes(options, model, model.graph_module, updated_model_args)  # type: ignore[return-value]

    @_beartype.beartype
    def pre_export_passes(
        self,
        options: exporter.ResolvedExportOptions,
        original_model: Union[torch.nn.Module, Callable],
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        # TODO: Import here to prevent circular dependency
        from torch.onnx._internal.fx import analysis, passes

        diagnostic_context = options.diagnostic_context

        # ONNX does not support concept of (implicit) type promotion.
        # Insert type casts explicitly where needed.
        fx_module = passes.InsertTypePromotion(diagnostic_context, fx_module).run()

        analysis.UnsupportedFxNodesAnalysis(
            diagnostic_context, fx_module, options.onnxfunction_dispatcher
        ).analyze(infra.levels.ERROR)

        # This operation should be invoked as the last pre export pass.
        # See [NOTE: Modularize pass ordering]
        fx_module = passes.Modularize(
            diagnostic_context, fx_module, is_exported_program=True
        ).run()

        return fx_module
