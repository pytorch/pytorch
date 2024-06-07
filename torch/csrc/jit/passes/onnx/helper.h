#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Utility functions for PyTorch to ONNX conversion.

static const int OPSET_VERSION_1 = 1;
static const int OPSET_VERSION_9 = 9;
static const int OPSET_VERSION_10 = 10;
static const int OPSET_VERSION_11 = 11;
static const int OPSET_VERSION_12 = 12;
static const int OPSET_VERSION_13 = 13;
static const int OPSET_VERSION_14 = 14;
static const int OPSET_VERSION_15 = 15;
static const int OPSET_VERSION_16 = 16;

using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

using ParamMap = std::map<std::string, IValue>;

TORCH_API void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);
TORCH_API ValueToParamPairMap
buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
TORCH_API void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);
TORCH_API void eraseUnusedBlockInputs(Block* b);
TORCH_API void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

TORCH_API Node* addNodeToBlock(
    Block* block,
    Symbol kind,
    ArrayRef<Value*> inputs);

TORCH_API Value* addInputToBlock(Block* block);

TORCH_API std::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type);

// Use int return type as no sable way exists to forward declare protobuf enum
TORCH_API int ATenTypeToOnnxType(at::ScalarType at_type);

TORCH_API void ONNXLintGraph(const std::shared_ptr<Graph>& graph);

Node* createONNXUnsqueeze(
    Graph* graph,
    Node* n_to_insert_before,
    Value* input,
    int axis,
    int opset_version);
Node* createONNXConstant(
    Graph* graph,
    Node* n_to_insert_before,
    at::Tensor value);

bool isValidToTransformToONNXConcatNode(Node* lc_node);

Node* transformToONNXConcatNode(
    Graph* graph,
    Node* lc_node,
    bool need_new_input,
    int opset_version);

class ScalarTypeHashFunction {
 public:
  size_t operator()(const c10::ScalarType& type) const {
    return static_cast<size_t>(type);
  }
};

} // namespace jit
} // namespace torch
