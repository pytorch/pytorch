#pragma once
#include <c10/macros/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

using ByteCodeEntry = std::tuple<std::string, IValue>;

TORCH_API void populate_upgraders_graph_map();

TORCH_API std::vector<ByteCodeEntry> generate_bytecode_list();

std::shared_ptr<Graph> create_upgrader_graph(
    const std::string& upgrader_name,
    const std::string& upgrader_body);

} // namespace jit
} // namespace torch
