"""
This script generates a CSV table with all ATen operators
supported by `torch.onnx.export`. The generated table is included by
docs/source/onnx_supported_aten_list.rst.
"""

import os

from torch.onnx import _onnx_supported_ops


# Constants
BUILD_DIR = "build/onnx"
SUPPORTED_OPS_CSV_FILE = "auto_gen_supported_op_list.csv"
UNSUPPORTED_OPS_CSV_FILE = "auto_gen_unsupported_op_list.csv"


def _sort_key(namespaced_opname):
    return tuple(reversed(namespaced_opname.split("::")))


def _get_op_lists():
    all_schemas = _onnx_supported_ops.all_forward_schemas()
    symbolic_schemas = _onnx_supported_ops.all_symbolics_schemas()
    supported_result = set()
    not_supported_result = set()
    for opname in all_schemas:
        if opname.endswith("_"):
            opname = opname[:-1]
        if opname in symbolic_schemas:
            # Supported op
            opsets = symbolic_schemas[opname].opsets
            supported_result.add((opname, f"Since opset {opsets[0]}"))
        else:
            # Unsupported op
            not_supported_result.add((opname, "Not yet supported"))
    return (
        sorted(supported_result, key=lambda x: _sort_key(x[0])),
        sorted(not_supported_result),
    )


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    supported, unsupported = _get_op_lists()

    with open(os.path.join(BUILD_DIR, SUPPORTED_OPS_CSV_FILE), "w") as f:
        f.write("Operator,opset_version(s)\n")
        for name, opset_version in supported:
            f.write(f'"``{name}``","{opset_version}"\n')

    with open(os.path.join(BUILD_DIR, UNSUPPORTED_OPS_CSV_FILE), "w") as f:
        f.write("Operator,opset_version(s)\n")
        for name, opset_version in unsupported:
            f.write(f'"``{name}``","{opset_version}"\n')


if __name__ == "__main__":
    main()
