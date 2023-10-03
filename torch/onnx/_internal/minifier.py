from __future__ import annotations

import functools
import logging
import re
from typing import List, Optional, Tuple

import onnxruntime

import torch
import torch.fx
from torch._functorch import fx_minifier
from torch.onnx._internal import exporter

logger = logging.getLogger(__name__)


def validate_in_ort(export_output, updated_model_args):
    # TODO: model larger than 2GB will require serializing to disk and reloading
    session = onnxruntime.InferenceSession(
        export_output.model_proto.SerializeToString()
    )
    session_inputs = {
        session_input.name: arg.cpu().numpy()
        for session_input, arg in zip(session.get_inputs(), updated_model_args)
    }
    session.run(None, session_inputs)


def module_fails_export_or_runtime(
    dynamo_exporter,
    graph_module,
    updated_model_args,
    expected_error_regex: Optional[str] = None,
) -> bool:
    logger.info(graph_module.print_readable(print_output=False))
    logger.info("Trying above graph.")
    try:
        export_output = dynamo_exporter._export_onnx_from_graph_module(graph_module)
        validate_in_ort(export_output, updated_model_args)
    except Exception as e:
        logger.info("Got Error: %s", e)
        if expected_error_regex is not None:
            if re.search(expected_error_regex, str(e)):
                return True
            else:
                # Some tiny minified graphs exhibit other unrelated errors
                # Filter them for now by matching only expected error.
                return False
        else:
            return True
    return False


def minifier(
    model,
    *args,
    expected_error_regex: Optional[str] = None,
    options: Optional[exporter.ExportOptions] = None,
    save_minified_repro_to_direrctory: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
    options = exporter.ResolvedExportOptions(None)
    # Keep graph_module flattened to enable it to be minified.
    options.modularization = False

    dynamo_exporter = exporter.Exporter(
        options=options,
        model=model,
        model_args=args,
        model_kwargs=kwargs,
    )

    graph_module, updated_model_args = dynamo_exporter._export_fx_graph_module()
    module_fails = functools.partial(
        module_fails_export_or_runtime,
        dynamo_exporter,
        expected_error_regex=expected_error_regex,
    )

    minified_graph_module, minified_inputs = fx_minifier.minifier(
        graph_module,
        updated_model_args,
        module_fails=module_fails,
        # save_dir="./",
        # offload_to_disk=True,
    )

    if save_minified_repro_to_direrctory is not None:
        minified_graph_module.to_folder(
            save_minified_repro_to_direrctory, "ReproModule"
        )
        torch.save(
            minified_graph_module.state_dict(),
            f"{save_minified_repro_to_direrctory}/minified_state_dict.pt",
        )
        torch.save(
            minified_inputs, f"{save_minified_repro_to_direrctory}/minified_inputs.pt"
        )

    return minified_graph_module, minified_inputs
