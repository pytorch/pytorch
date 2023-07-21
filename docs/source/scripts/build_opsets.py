import os
from pathlib import Path

import torch
from collections import OrderedDict
from torchgen.gen import parse_native_yaml
import torch._prims as prims
from torch._decomp import decomposition_table

ROOT = Path(__file__).absolute().parent.parent.parent.parent
NATIVE_FUNCTION_YAML_PATH = ROOT / Path("aten/src/ATen/native/native_functions.yaml")
TAGS_YAML_PATH = ROOT / Path("aten/src/ATen/native/tags.yaml")

BUILD_DIR = "build/ir"
ATEN_OPS_CSV_FILE = "aten_ops.csv"
PRIMS_OPS_CSV_FILE = "prims_ops.csv"


def get_aten():
    parsed_yaml = parse_native_yaml(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH)
    native_functions = parsed_yaml.native_functions

    aten_ops = OrderedDict()
    for function in native_functions:
        if "core" in function.tags:
            op_name = str(function.func.name)
            aten_ops[op_name] = function

    op_schema_pairs = []
    for key, op in sorted(aten_ops.items()):
        aten_full_name = f"aten.{key}"
        schema = str(op.func).replace("*", r"\*")

        op_name_overload = key.split(".")
        op_name = op_name_overload[0]
        if len(op_name_overload) == 2:
            overload = op_name_overload[1]
        else:
            overload = "default"

        op_overloadpacket = getattr(torch.ops.aten, op_name, None)
        op_overload = getattr(op_overloadpacket, overload, None)
        decomp_fn = decomposition_table.get(op_overload, None)
        has_decomp = decomp_fn is not None and decomp_fn.__module__ in {"torch._decomp.decompositions"}

        op_schema_pairs.append((aten_full_name, schema, has_decomp))

    return op_schema_pairs


def get_prims():
    op_schema_pairs = []
    for op_name in prims.__all__:

        op_overload = getattr(prims, op_name, None)

        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        op_overloadpacket = op_overload.overloadpacket

        op_name = str(op_overload).replace(".default", "")
        schema = op_overloadpacket.schema.replace("*", r"\*")

        op_schema_pairs.append((op_name, schema))

    return op_schema_pairs

def main():
    aten_ops_list = get_aten()
    prims_ops_list = get_prims()

    os.makedirs(BUILD_DIR, exist_ok=True)

    with open(os.path.join(BUILD_DIR, ATEN_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema,Has Decomp\n")
        for name, schema, has_decomp in aten_ops_list:
            f.write(f'"``{name}``","{schema}","{has_decomp}"\n')

    with open(os.path.join(BUILD_DIR, PRIMS_OPS_CSV_FILE), "w") as f:
        f.write("Operator,Schema\n")
        for name, schema in prims_ops_list:
            f.write(f'"``{name}``","{schema}"\n')


if __name__ == '__main__':
    main()
