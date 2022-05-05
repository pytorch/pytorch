#!/usr/bin/env python3
import os
from pathlib import Path

from torch.jit._shape_functions import shape_compute_graph_mapping

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
#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>
#include <torch/csrc/jit/runtime/operator.h>

// clang-format off

namespace torch {
namespace jit {


std::string shape_funcs = ""
"""


DECOMP_CENTER = r"""


const std::string& GetSerializedShapeFunctions() {
  return shape_funcs;
}

const OperatorMap<std::string>& GetShapeFunctionMappings() {
 static const OperatorMap<std::string> shape_mappings {
"""

DECOMP_END = r"""
  };

  return shape_mappings;
}

// clang-format on

} // namespace jit
} // namespace torch
"""


SERIALIZED_SHAPE_UTIL_FILE_NAME = "serialized_shape_function_registry.cpp"


def gen_serialized_decompisitions() -> str:
    already_serialized_names = set()
    unique_funcs = []
    for scripted_func in shape_compute_graph_mapping.values():
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


def gen_shape_mappings() -> str:
    shape_mappings = []
    for schema, scripted_func in shape_compute_graph_mapping.items():
        shape_mappings.append('    {"' + schema + '", "' + scripted_func.name + '"},')
    return "\n".join(shape_mappings)


def write_decomposition_util_file(path: str) -> None:
    decomposition_str = gen_serialized_decompisitions()
    shape_mappings = gen_shape_mappings()
    file_components = [
        SHAPE_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        shape_mappings,
        DECOMP_END,
    ]
    print("writing file to : ", path + "/" + SERIALIZED_SHAPE_UTIL_FILE_NAME)
    with open(os.path.join(path, SERIALIZED_SHAPE_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))


def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[3]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))


if __name__ == "__main__":
    main()
