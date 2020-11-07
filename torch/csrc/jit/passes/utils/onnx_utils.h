#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace ONNX_NAMESPACE {
enum TensorProto_DataType : int;
}

namespace torch {
namespace jit {

// Utility functions for PyTorch to ONNX conversion.

static const int OPSET_VERSION_1 = 1;
static const int OPSET_VERSION_9 = 9;
static const int OPSET_VERSION_10 = 10;
static const int OPSET_VERSION_11 = 11;
static const int OPSET_VERSION_12 = 12;

using ValueToParamPairMap = std::map<Value*, std::pair<std::string, IValue>>;

using ParamMap = std::map<std::string, IValue>;

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);
ValueToParamPairMap buildValueToParamsMap(Block* b, const ParamMap& paramsDict);
void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap);
void eraseUnusedBlockInputs(Block* b);
void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict);

Node* addNodeToBlock(Block* block, Symbol kind, ArrayRef<Value*> inputs);

Value* addInputToBlock(Block* block);

TORCH_API c10::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type);

TORCH_API ONNX_NAMESPACE::TensorProto_DataType ATenTypeToOnnxType(
    at::ScalarType at_type);

} // namespace jit
} // namespace torch
