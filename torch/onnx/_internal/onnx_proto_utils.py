"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""

import io
import os
import zipfile
from typing import Mapping, Union

import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors  # noqa: F401
from torch.onnx._internal import _beartype, jit_utils, registration


@_beartype.beartype
def _export_file(
    model_bytes: bytes,
    f: Union[io.BytesIO, str],
    export_type: str,
    export_map: Mapping[str, bytes],
) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    # TODO(titaiwang) MYPY asks for os.PathLike[str] type for parameter: f,
    # but beartype raises beartype.roar.BeartypeDecorHintNonpepException,
    # as os.PathLike[str] uncheckable at runtime
    if export_type == _exporter_states.ExportTypes.PROTOBUF_FILE:
        assert len(export_map) == 0
        with torch.serialization._open_file_like(f, "wb") as opened_file:
            opened_file.write(model_bytes)
    elif export_type in [
        _exporter_states.ExportTypes.ZIP_ARCHIVE,
        _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
    ]:
        compression = (
            zipfile.ZIP_DEFLATED
            if export_type == _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE
            else zipfile.ZIP_STORED
        )
        with zipfile.ZipFile(f, "w", compression=compression) as z:
            z.writestr(_constants.ONNX_ARCHIVE_MODEL_PROTO_NAME, model_bytes)
            for k, v in export_map.items():
                z.writestr(k, v)
    elif export_type == _exporter_states.ExportTypes.DIRECTORY:
        if os.path.exists(f):  # type: ignore[arg-type]
            assert os.path.isdir(f)  # type: ignore[arg-type]
        else:
            os.makedirs(f)  # type: ignore[arg-type]

        model_proto_file = os.path.join(f, _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME)  # type: ignore[arg-type]
        with torch.serialization._open_file_like(model_proto_file, "wb") as opened_file:
            opened_file.write(model_bytes)

        for k, v in export_map.items():
            weight_proto_file = os.path.join(f, k)  # type: ignore[arg-type]
            with torch.serialization._open_file_like(
                weight_proto_file, "wb"
            ) as opened_file:
                opened_file.write(v)
    else:
        raise RuntimeError("Unknown export type")


@_beartype.beartype
def _add_onnxscript_fn(
    model_bytes: bytes,
    custom_opsets: Mapping[str, int],
) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""

    # TODO(titaiwang): remove this when onnx becomes dependency
    try:
        import onnx
    except ImportError:
        raise errors.OnnxExporterError("Module onnx is not installed!")

    # For > 2GB model, onnx.load_fromstring would fail. However, because
    # in _export_onnx, the tensors should be saved separately if the proto
    # size > 2GB, and if it for some reason did not, the model would fail on
    # serialization anyway in terms of the protobuf limitation. So we don't
    # need to worry about > 2GB model getting here.
    model_proto = onnx.load_from_string(model_bytes)

    # Iterate graph nodes to insert only the included custom
    # function_proto into model_proto
    # TODO(titaiwang): Currently, onnxscript doesn't support ONNXFunction
    # calling other ONNXFunction scenario, neither does it here
    onnx_function_list = list()
    included_node_func = set()
    for node in model_proto.graph.node:
        node_kind = node.domain + "::" + node.op_type
        if (
            jit_utils.is_custom_domain(node.domain)
            and node_kind not in included_node_func
        ):
            specified_version = custom_opsets.get(node.domain, 1)
            onnx_function_group = registration.registry.get_function_group(node_kind)
            if onnx_function_group is not None:
                onnx_fn = onnx_function_group.get(specified_version)
                if onnx_fn is not None:
                    # TODO(titaiwang): to_function_proto is onnx-script API and can be annotated
                    # after onnx-script is dependency
                    onnx_function_list.append(onnx_fn.to_function_proto())  # type: ignore[attr-defined]
                    included_node_func.add(node_kind)
                    continue

            raise errors.UnsupportedOperatorError(
                node_kind,
                specified_version,
                onnx_function_group.get_min_supported()
                if onnx_function_group
                else None,
            )
    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()
    return model_bytes
