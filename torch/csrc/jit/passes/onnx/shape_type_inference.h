#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>

namespace torch {
namespace jit {

TORCH_API TypePtr
MergeInferredType(TypePtr existing_type, TypePtr inferred_type);

// Update graph input types with dynamic axes info.
// Axes that are marked as dynamic will be assigned as dynamic ShapeSymbol.
// Note it is possible for multiple axes to share the same ShapeSymbol,
// if they are defined as such in dynamic_axes.
TORCH_API void ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    const std::vector<std::string>& input_names);

// Update graph output with types of output Tensors.
// If onnx_shape_inference is true, types of output Tensors will be compared and
// merged with inferred types. It is possible that inferred types contain
// dynamic axes, hence it takes precedence over types of output Tensors.
TORCH_API void ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    const python::IODescriptor& desc,
    bool onnx_shape_inference);

// Utilize ONNX Shape Inference for node.
// The node must have ONNX namespace, and is valid ONNX node accroding to spec.
// On successful ONNX shape inference runs, the function updates output types of
// n with inferred shape and type. Otherwise n is unchanged.
TORCH_API void ONNXShapeTypeInference(
    Node* n,
    const ParamMap& params_dict,
    int opset_version);

// Utilize ONNX Shape Inference for graph.
// Internally calls ONNXShapeTypeInference for each node, to achieve more
// coverage that skips only individual nodes if illegal, instead of skipping for
// the entire graph.
TORCH_API void ONNXShapeTypeInference(
    std::shared_ptr<Graph>& g,
    const ParamMap& params_dict,
    int opset_version);

} // namespace jit
} // namespace torch
