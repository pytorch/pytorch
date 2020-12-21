#include <torch/csrc/jit/passes/onnx/helper.h>
#include <onnx/onnx_pb.h>

namespace torch {
namespace jit {
namespace onnx {
using namespace ::c10::onnx;

}

ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->debugName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

c10::optional<at::ScalarType> ONNXTypeToATenType(int32_t onnx_type) {
  switch (onnx_type) {
    case ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return at::ScalarType::Undefined;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return at::kByte;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return at::kChar;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return at::kShort;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return at::kInt;
    case ::ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return at::kLong;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return at::kBool;
    case ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case ::ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
      return at::kComplexFloat;
    case ::ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      return at::kComplexDouble;
    case ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return at::kBFloat16;
    default:
      TORCH_CHECK("unexpected tensor scalar type");
  }
  return c10::optional<at::ScalarType>{};
}

Node* addNodeToBlock(Block* block, Symbol kind, ArrayRef<Value*> inputs) {
  auto new_node = block->appendNode(block->owningGraph()->create(kind));
  for (auto input : inputs) {
    auto new_input = new_node->addInput(input);
  }
  return new_node;
}

Value* addInputToBlock(Block* block) {
  return block->addInput();
}

} // namespace jit
} // namespace torch
