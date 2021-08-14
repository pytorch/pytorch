#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This api will be used by serialization/export.cpp to extract function
// information. It should do conversion on graph to
//    1. Extract and preserve functions definition.
//    2. Replace nodes within functions with a single node reflecting that
//    function type.
// export.cpp will convert to function_proto and node of function kind with
// above information.
//
// Possible different designs to achieve the above 2 points.
//    1. Return a separate list of <name, graph> pairs representing functions.
//    2. Return a single graph, with dummy node within that keep function as
//    subgraph.
//
// More details
//    1. Manage inputs/initializers/attributes/constants of function.

TORCH_API std::pair<std::unordered_map<const Value*, std::string>, std::unordered_map<const Node*, std::unordered_map<std::string, std::string>>> ONNXFunctionExtraction(std::shared_ptr<Graph>& graph, const std::vector<std::string>& module_names, const std::vector<std::string>& param_names);

} // namespace jit
} // namespace torch
