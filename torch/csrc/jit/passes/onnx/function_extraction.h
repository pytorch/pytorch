#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This api will be used by serialization/export.cpp to extract function
// information. It should do conversion on graph to
//    1. Extract and preserve functions definition.
//    2. Replace nodes within functions with a single node reflecting that
//    function type.
// Function attribute map information is also returned, as Torch IR cannot
// represent these info inside Graph object.
// export.cpp will serialize the ONNX model with function_proto with
// above information.
namespace onnx {

using ValAttrNameMap = std::unordered_map<const Value*, std::string>;
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

TORCH_API std::pair<ValAttrNameMap, NodeAttrNameMap> ONNXFunctionExtraction(
    std::shared_ptr<Graph>& graph,
    const std::vector<std::string>& module_names,
    const std::vector<std::string>& param_names);
} // namespace onnx

} // namespace jit
} // namespace torch
