#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This api will be used by serialization/export.cpp to extract function
// information. It should do conversion on graph to
//    1. Extract subgraph pattern of functions and define as local function
//    node.
//    2. Replace subgraph pattern of functions with a single node reflecting
//    that local function node type.
// Function attribute map information is also returned, as Torch IR cannot
// represent these info inside Graph object.
// export.cpp will serialize the ONNX model with function_proto with
// above information.
namespace onnx {

// The following return types are used to track information regarding function
// attributes, that are unable to be traced through Torch IR.
// NodeAttrNameMap tracks mapping from attribute name of IR Node inside function
// subgraph, to function attribute name. Here's an example of exporting CELU and
// LayerNorm.
//
// clang-format off
// class M(torch.nn.Module):
//     def __init__(self) -> None:
//         super().__init__()
//         self.lns = torch.nn.ModuleList([torch.nn.LayerNorm(3, eps = i) for i in range(2)])
//         self.celu1 = torch.nn.CELU(1.0)
//         self.celu2 = torch.nn.CELU(2.0)

//     def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
//         res1 = self.celu1(x)
//         res2 = self.celu2(y)
//         for ln in self.lns:
//             z = ln(z)
//         return res1 + res2 + z
// clang-format on
//
// Returning
//
// NodeAttrNameMap:
// {
//    %1 : Float(2, 3) = onnx::Celu[alpha=2.](%y) : {
//      'alpha' : 'Celu_alpha'
//    }
// }
//
// The info here helps graph._export_onnx to construct function attributes for
// onnx local FunctionProto.
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

TORCH_API NodeAttrNameMap ONNXFunctionExtraction(
    std::shared_ptr<Graph>& graph,
    const std::unordered_set<std::string>& module_names,
    const std::vector<std::string>& param_names);

TORCH_API void ONNXClearScopeRecords();

TORCH_API void ONNXTrackScopeAttributes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& attributes);

} // namespace onnx

} // namespace jit
} // namespace torch
