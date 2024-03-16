"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""

from __future__ import annotations

import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union

import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration


@_beartype.beartype
def export_as_test_case(
    model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str
) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Export test case to ONNX format failed: Please install ONNX."
        ) from exc

    test_case_dir = os.path.join(dir, "test_" + name)
    os.makedirs(test_case_dir, exist_ok=True)
    _export_file(
        model_bytes,
        os.path.join(test_case_dir, "model.onnx"),
        _exporter_states.ExportTypes.PROTOBUF_FILE,
        {},
    )
    data_set_dir = os.path.join(test_case_dir, "test_data_set_0")
    if os.path.exists(data_set_dir):
        shutil.rmtree(data_set_dir)
    os.makedirs(data_set_dir)

    proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    for i, (input_proto, input) in enumerate(zip(proto.graph.input, inputs_data)):
        export_data(input, input_proto, os.path.join(data_set_dir, f"input_{i}.pb"))
    for i, (output_proto, output) in enumerate(zip(proto.graph.output, outputs_data)):
        export_data(output, output_proto, os.path.join(data_set_dir, f"output_{i}.pb"))

    return test_case_dir


@_beartype.beartype
def load_test_case(dir: str) -> Tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        model_bytes: The ONNX model in bytes.
        inputs: the inputs data, mapping from input name to numpy.ndarray.
        outputs: the outputs data, mapping from output name to numpy.ndarray.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError(
            "Load test case from ONNX format failed: Please install ONNX."
        ) from exc

    with open(os.path.join(dir, "model.onnx"), "rb") as f:
        model_bytes = f.read()

    test_data_dir = os.path.join(dir, "test_data_set_0")

    inputs = {}
    input_files = glob.glob(os.path.join(test_data_dir, "input_*.pb"))
    for input_file in input_files:
        tensor = onnx.load_tensor(input_file)  # type: ignore[attr-defined]
        inputs[tensor.name] = numpy_helper.to_array(tensor)
    outputs = {}
    output_files = glob.glob(os.path.join(test_data_dir, "output_*.pb"))
    for output_file in output_files:
        tensor = onnx.load_tensor(output_file)  # type: ignore[attr-defined]
        outputs[tensor.name] = numpy_helper.to_array(tensor)

    return model_bytes, inputs, outputs


@_beartype.beartype
def export_data(data, value_info_proto, f: str) -> None:
    """Export data to ONNX protobuf format.

    Args:
        data: The data to export, nested data structure of numpy.ndarray.
        value_info_proto: The ValueInfoProto of the data. The type of the ValueInfoProto
            determines how the data is stored.
        f: The file to write the data to.
    """
    try:
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError(
            "Export data to ONNX format failed: Please install ONNX."
        ) from exc

    with open(f, "wb") as opened_file:
        if value_info_proto.type.HasField("map_type"):
            opened_file.write(
                numpy_helper.from_dict(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("sequence_type"):
            opened_file.write(
                numpy_helper.from_list(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("optional_type"):
            opened_file.write(
                numpy_helper.from_optional(
                    data, value_info_proto.name
                ).SerializeToString()
            )
        else:
            assert value_info_proto.type.HasField("tensor_type")
            opened_file.write(
                numpy_helper.from_array(data, value_info_proto.name).SerializeToString()
            )


@_beartype.beartype
def _export_file(
    model_bytes: bytes,
    f: Union[io.BytesIO, str],
    export_type: str,
    export_map: Mapping[str, bytes],
) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    if export_type == _exporter_states.ExportTypes.PROTOBUF_FILE:
        assert len(export_map) == 0
        with torch.serialization._open_file_like(f, "wb") as opened_file:
            opened_file.write(model_bytes)
    elif export_type in {
        _exporter_states.ExportTypes.ZIP_ARCHIVE,
        _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
    }:
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
        if isinstance(f, io.BytesIO) or not os.path.isdir(f):  # type: ignore[arg-type]
            raise ValueError(
                f"f should be directory when export_type is set to DIRECTORY, instead get type(f): {type(f)}"
            )
        if not os.path.exists(f):  # type: ignore[arg-type]
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
        raise ValueError("Unknown export type")


@_beartype.beartype
def _add_onnxscript_fn(
    model_bytes: bytes,
    custom_opsets: Mapping[str, int],
) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""
    try:
        import onnx
    except ImportError as e:
        raise errors.OnnxExporterError("Module onnx is not installed!") from e

    # For > 2GB model, onnx.load_fromstring would fail. However, because
    # in _export_onnx, the tensors should be saved separately if the proto
    # size > 2GB, and if it for some reason did not, the model would fail on
    # serialization anyway in terms of the protobuf limitation. So we don't
    # need to worry about > 2GB model getting here.
    model_proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    # Iterate graph nodes to insert only the included custom
    # function_proto into model_proto
    onnx_function_list = list()  # type: ignore[var-annotated]
    included_node_func = set()  # type: Set[str]
    # onnx_function_list and included_node_func are expanded in-place
    _find_onnxscript_op(
        model_proto.graph, included_node_func, custom_opsets, onnx_function_list
    )

    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()
    return model_bytes


@_beartype.beartype
def _find_onnxscript_op(
    graph_proto,
    included_node_func: Set[str],
    custom_opsets: Mapping[str, int],
    onnx_function_list: List,
):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
    for node in graph_proto.node:
        node_kind = node.domain + "::" + node.op_type
        # Recursive needed for control flow nodes: IF/Loop which has inner graph_proto
        for attr in node.attribute:
            if attr.g is not None:
                _find_onnxscript_op(
                    attr.g, included_node_func, custom_opsets, onnx_function_list
                )
        # Only custom Op with ONNX function and aten with symbolic_fn should be found in registry
        onnx_function_group = registration.registry.get_function_group(node_kind)
        # Ruled out corner cases: onnx/prim in registry
        if (
            node.domain
            and not jit_utils.is_aten(node.domain)
            and not jit_utils.is_prim(node.domain)
            and not jit_utils.is_onnx(node.domain)
            and onnx_function_group is not None
            and node_kind not in included_node_func
        ):
            specified_version = custom_opsets.get(node.domain, 1)
            onnx_fn = onnx_function_group.get(specified_version)
            if onnx_fn is not None:
                if hasattr(onnx_fn, "to_function_proto"):
                    onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                    onnx_function_list.append(onnx_function_proto)
                    included_node_func.add(node_kind)
                continue

            raise errors.UnsupportedOperatorError(
                node_kind,
                specified_version,
                onnx_function_group.get_min_supported()
                if onnx_function_group
                else None,
            )
    return onnx_function_list, included_node_func
