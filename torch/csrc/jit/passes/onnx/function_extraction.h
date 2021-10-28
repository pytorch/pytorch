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

// The following return types are used to track information regarding function
// attributes, that are unable to be traced through Torch IR.
// ValAttrNameMap tracks mapping from IR Value inside function subgraph,
// to function attribute name.
// NodeAttrNameMap tracks mapping from attribute name of IR Node inside function
// subgraph, to function attribute name. Here's an example of exporting CELU and
// LayerNorm.
//
// clang-format off
// class M(torch.nn.Module):
//     def __init__(self):
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
// ValAttrNameMap:
// {
//    %8 : Float = onnx::Constant[value={1}]() : 'Constant_25'
// }
// NodeAttrNameMap:
// {
//    %1 : Float(2, 3) = onnx::Celu[alpha=2.](%y) : {
//      'alpha' : 'Celu_alpha'
//    }
// }
//
// The info here helps graph._export_onnx to construct function attributes for
// onnx local FunctionProto.
using ValAttrNameMap = std::unordered_map<const Value*, std::string>;
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

TORCH_API std::pair<ValAttrNameMap, NodeAttrNameMap> ONNXFunctionExtraction(
    std::shared_ptr<Graph>& graph,
    const std::unordered_set<std::string>& module_names,
    const std::vector<std::string>& param_names);
} // namespace onnx

} // namespace jit
} // namespace torch
