#!/usr/bin/env python3
import os
from pathlib import Path

from torch.jit._decompositions import decomposition_table

# from torchgen.code_template import CodeTemplate

DECOMP_HEADER = r"""
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>

namespace torch {
namespace jit {


const std::string decomp_funcs =
R"("""


DECOMP_CENTER = r"""
)";

const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
"""

DECOMP_END = r"""
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace jit
} // namespace torch
"""


DECOMPOSITION_UTIL_FILE_NAME = "decomposition_registry_util.cpp"


def gen_serialized_decompisitions() -> str:
    return "\n".join(
        [scripted_func.code for scripted_func in decomposition_table.values()]  # type: ignore[misc]
    )


def gen_decomposition_mappings() -> str:
    decomposition_mappings = []
    for schema, scripted_func in decomposition_table.items():
        decomposition_mappings.append(
            '    {"' + schema + '", "' + scripted_func.name + '"},'  # type: ignore[operator]
        )
    return "\n".join(decomposition_mappings)


def write_decomposition_util_file(path: str) -> None:
    decomposition_str = gen_serialized_decompisitions()
    decomposition_mappings = gen_decomposition_mappings()
    file_components = [
        DECOMP_HEADER,
        decomposition_str,
        DECOMP_CENTER,
        decomposition_mappings,
        DECOMP_END,
    ]
    print("writing file to : ", path + "/" + DECOMPOSITION_UTIL_FILE_NAME)
    with open(os.path.join(path, DECOMPOSITION_UTIL_FILE_NAME), "wb") as out_file:
        final_output = "".join(file_components)
        out_file.write(final_output.encode("utf-8"))


def main() -> None:
    pytorch_dir = Path(__file__).resolve().parents[3]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "runtime"
    write_decomposition_util_file(str(upgrader_path))


if __name__ == "__main__":
    main()
