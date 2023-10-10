# NOTE: This file is referenced by name at
#       /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES.
#       introduced by https://github.com/pytorch/pytorch/pull/98894.
#       If this file is renamed, moved, etc please update the reference there!

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter


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
        model: "ExportedProgram",  # type: ignore[name-defined]
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
        return self.pre_export_passes(options, model, model.graph_module, model_args)  # type: ignore[return-value]

    @_beartype.beartype
    def pre_export_passes(
        self,
        options: exporter.ResolvedExportOptions,
        original_model: Union[torch.nn.Module, Callable],
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        return fx_module
