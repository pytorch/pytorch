#include <torch/csrc/jit/passes/onnx/eliminate_unused_items.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
 
#include <c10/util/Optional.h>
#include <algorithm>
 
namespace torch {
namespace jit {
 
namespace onnx {
using namespace ::c10::onnx;
}
 
namespace {
 
void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}
 
void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}
}
 
void EliminateUnusedItemsONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}
 
} // namespace jit
} // namespace torch