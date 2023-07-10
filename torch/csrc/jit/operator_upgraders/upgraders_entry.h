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

TORCH_API void populate_upgraders_graph_map();

TORCH_API std::unordered_map<std::string, std::shared_ptr<Graph>>
generate_upgraders_graph();

TORCH_API std::unordered_map<std::string, std::string> get_upgraders_entry_map();

std::shared_ptr<Graph> create_upgrader_graph(
    const std::string& upgrader_name,
    const std::string& upgrader_body);

} // namespace jit
} // namespace torch
