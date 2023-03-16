from __future__ import annotations

import contextlib
import inspect

from typing import Any, Callable, List, Tuple, Union

import onnx

import torch
import torch._C
import torch._decomp
import torch._dynamo
import torch._ops
import torch.fx

from torch.onnx import _constants

from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import frontend, function_dispatcher, options, passes
from torch.utils import _pytree


@contextlib.contextmanager
def patch_pytree_huggingface_modeloutput():
    """
    Patch 'torch.utils._pytree' to support family of 'ModelOutput' from HuggingFace 'transformers'.

    The source and details of the issue is described at https://github.com/pytorch/pytorch/issues/96386.
    This patch enables 'torch.utils._pytree' to flatten and unflatten all 'ModelOutput'
    subclasses defined in HuggingFace 'transformers'. Hence resolving the mismatch between
    `dynamo` eager traced model outputs and `dynamo.export` fx graph computed outputs.

    FIXME(bowbao): Remove this patch after above issue is resolved for `dynamo.export`.
    """
    try:
        from transformers import modeling_outputs  # type: ignore[import]
    except ImportError as e:
        # Do nothing if 'transformers' is not installed.
        try:
            yield
        finally:
            pass
        return

    def model_output_flatten(
        output: modeling_outputs.ModelOutput,
    ) -> Tuple[List[Any], _pytree.Context]:
        return list(output.values()), (type(output), list(output.keys()))

    def model_output_unflatten(
        values: List[Any], context: _pytree.Context
    ) -> modeling_outputs.ModelOutput:
        output_type, keys = context
        return output_type(**dict(zip(keys, values)))

    # All 'ModelOutput' subclasses are defined under module 'modeling_outputs'.
    named_model_output_classes = inspect.getmembers(
        modeling_outputs,
        lambda x: inspect.isclass(x) and issubclass(x, modeling_outputs.ModelOutput),
    )

    for _, class_type in named_model_output_classes:
        _pytree._register_pytree_node(
            class_type, model_output_flatten, model_output_unflatten
        )

    try:
        yield
    finally:
        for _, class_type in named_model_output_classes:
            _pytree.SUPPORTED_NODES.pop(class_type)


@_beartype.beartype
def _export(
    module: torch.fx.GraphModule,
    *args,
    **kwargs,
) -> Union["onnx.ModelProto", bytes]:
    export_options = options.ExportOptions()
    export_options.update(**kwargs)
    # Apply decomposition table to the input graph.
    # Make sure the feed-in "module" is stateless.
    # Ensure placeholder targets match the original module's signature since
    # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
    decomposed_module = passes.Decompose(
        module, export_options.decomposition_table
    ).run(*args)
    # Run FakeTensorProp on decomposed_module.
    # Symbolic output of the i-th node can be accessed via
    # decomposed_module.graph.nodes[i].meta["val"]
    decomposed_module = passes.ShapeInferenceWithFakeTensor(decomposed_module).run(
        *args
    )

    # We want to pass list of ints and floats to TorchScript graph correctly
    # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
    # receive FakeTensor and results runtime error. In addition, TorchScript-based
    # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
    # with FakeTensorMode.
    with torch.utils._mode_utils.no_dispatch():
        onnxscript_graph = passes.export_fx_to_onnxscript(
            decomposed_module, export_options
        )
    # Export TorchScript graph to ONNX ModelProto.
    onnx_model = onnxscript_graph.to_model_proto(export_options.opset_version)

    if export_options.use_binary_format:
        # Return ModelProto in binary format.
        return onnx_model.SerializeToString()
    # Return ModelProto
    return onnx_model


@_beartype.beartype
@patch_pytree_huggingface_modeloutput()
def export(
    fn: Union[torch.nn.Module, Callable],
    *args,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
    **kwargs,
) -> Union["onnx.ModelProto", bytes]:
    # Translate callable to FX graph.
    # TODO(bowbao, titai): Change "real" to "symbolic" after symbolic shape export is supported.
    fx_frontend = frontend.DynamoExport(tracing_mode="real", aten_graph=True)
    graph_module = fx_frontend.trace(fn, *args)
    # Export FX graph to ONNX ModelProto.
    #
    # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
    # to _export.
    return _export(
        graph_module,
        *(args + tuple(kwargs.values())),
        opset_version=opset_version,
        decomposition_table=function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
        op_level_debug=op_level_debug,
    )
