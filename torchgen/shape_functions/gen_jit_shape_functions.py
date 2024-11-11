#!/usr/bin/env python3
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from pathlib import Path


# Manually importing the shape function module based on current directory
# instead of torch imports to avoid needing to recompile Pytorch before
# running the script

file_path = Path.cwd() / "torch" / "jit" / "_shape_functions.py"
module_name = "torch.jit._shape_functions"

err_msg = """Could not find shape functions file, please make sure
you are in the root directory of the Pytorch git repo"""
if not file_path.exists():
    raise Exception(err_msg)  # noqa: TRY002

spec = spec_from_file_location(module_name, file_path)
assert spec is not None
module = module_from_spec(spec)
sys.modules[module_name] = module
assert spec.loader is not None
assert module is not None
spec.loader.exec_module(module)

bounded_compute_graph_mapping = module.bounded_compute_graph_mapping
shape_compute_graph_mapping = module.shape_compute_graph_mapping


SHAPE_HEADER = r"""
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python
 * torchgen/shape_functions/gen_jit_shape_functions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>

// clang-format off

namespace torch {
namespace jit {


std::string shape_funcs = ""
"""


DECOMP_CENTER = r"""


const std::string& GetSerializedShapeFunctions() {
  return shape_funcs;
}

"""

DECOMP_END = r"""
// clang-format on

} // namespace jit
} // namespace torch
"""


SERIALIZED_SHAPE_UTIL_FILE_NAME = "serialized_shape_function_registry.cpp"


def gen_serialized_decompisitions() -> str:
    already_serialized_names = set()
    unique_funcs = []
    all_funcs = chain(
        shape_compute_graph_mapping.values(), *bounded_compute_graph_mapping.values()
    )
    for scripted_func in all_funcs:
        if scripted_func.name in already_serialized_names:
            continue
        already_serialized_names.add(scripted_func.name)
        unique_funcs.append(scripted_func)

    output_strs = []
    curr_str = ""
    for scripted_func in unique_funcs:
        serialized_code = scripted_func.code
        # technically its higher but give a buffer bc there are weird rules
        # around some characters
        # TODO: this was the limit I found by googling but it seems way
        # too short ?
        MAX_MSFT_STR_LEN = 2000
        if len(curr_str) + len(serialized_code) <= MAX_MSFT_STR_LEN:
            curr_str += "\n" + serialized_code
        else:
            output_strs.append(curr_str)
            curr_str = scripted_func.code
    output_strs.append(curr_str)

    final_output = ""
    # Windows compiler doesnt correctly handle adjacent
    # string literals
    for output_str in output_strs:
        start = '+ std::string(R"=====('
        end = '\n)=====")\n'
        final_output += start + output_str + end
    final_output += ";"
    return final_output


SHAPE_SCHEMA_START = r"""
const OperatorMap<std::string>& GetShapeFunctionMappings() {
 static const OperatorMap<std::string> shape_mappings {
"""

SHAPE_SCHEMA_END = r"""
  };

  return shape_mappings;
}
"""


def gen_shape_mappings() -> str:
    shape_mappings = []
    for schema, scripted_func in shape_compute_graph_mapping.items():
        shape_mappings.append('    {"' + schema + '", "' + scripted_func.name + '"},')
    return SHAPE_SCHEMA_START + "\n".join(shape_mappings) + SHAPE_SCHEMA_END


BOUNDED_SCHEMA_START = r"""
const OperatorMap<std::pair<std::string, std::string>>& GetBoundedShapeMappings() {
 static const OperatorMap<std::pair<std::string, std::string>> shape_mappings {
"""


def gen_bounded_mappings() -> str:
    bounded_mappings = []
    for schema, (lower_func, upper_func) in bounded_compute_graph_mapping.items():
        map_str = (
            '    {"'
            + schema
            + '", {"'
            + lower_func.name
            + '", "'
            + upper_func.name
            + '"}},'
        )
        bounded_mappings.append(map_str)
    return BOUNDED_SCHEMA_START + "\n".join(bounded_mappings) + SHAPE_SCHEMA_END


def write_decomposition_util_file(path: str) -> None:
    decomposition_str = gen_serialized_decompisitions()
    shape_mappings = gen_shape_mappings()
    bounded_mappings = gen_bounded_mappings()
    file_components = [
        SHAPE_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        shape_mappings,
        bounded_mappings,
        DECOMP_END,
    ]
    print("writing file to : ", path + "/" + SERIALIZED_SHAPE_UTIL_FILE_NAME)
    with open(os.path.join(path, SERIALIZED_SHAPE_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))


def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[2]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))


if __name__ == "__main__":
    main()
