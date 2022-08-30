#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>

#include <utility>

namespace torch {
namespace jit {

// Merges existing_type and inferred_type.
// Returns {merged type, whether or not inferred_type was used}.
//
// The inferred type will take higher precedence, since it is produced by ONNX
// shape inference, and is more compatible with ONNX. In cases where ONNX shape
// inference fails to produce an inferred type, or produces an inferred type
// that is incomplete, refer to existing type and fill in the gap that is
// missing. Currently the following cases are supported.
//  1. existing type: Tensor[], inferred type: Tensor[]
//    For list of tensors, existing type does not store datatype nor shape for
//    inner tensor. Thus inferred type always contain more information, and is
//    returned.
//  2. existing type: Tensor, inferred type: Tensor
//    Fill in missing info (shape, data type) for inferred type from existing
//    type.
//  3. existing type: Scalar[], inferred type: Tensor
//    ONNX represents list of scalars by 1-d Tensor. Return inferred type since
//    it is more compatible with ONNX.
std::pair<TypePtr, bool> MergeInferredType(
    TypePtr existing_type,
    TypePtr inferred_type);

void MergeInferredTypeAndSetMap(
    Value* dest_v,
    TypePtr existing_type,
    TypePtr inferred_type,
    bool set_constant_value_map = true);

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
    bool onnx_shape_inference,
    bool is_script);

// Utilize ONNX Shape Inference for node.
// The node must have ONNX namespace, and is valid ONNX node according to spec.
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

std::pair<bool, bool> AreInputsReliableOrStatic(Node* n);
void UpdateReliable(
    torch::jit::Value* output,
    const std::pair<bool, bool>& input_reliable);

void UpdateReliable(torch::jit::Node* n);

} // namespace jit
} // namespace torch
